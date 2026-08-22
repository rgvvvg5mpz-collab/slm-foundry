"""Shared trainer plumbing: the request, the result, and the context object.

Trainers never touch the database. They are handed a :class:`TrainRequest` of
plain data and a :class:`TrainContext` with three verbs — ``log``, ``metric``,
``checkpoint`` — and the worker maps those onto job events, heartbeats, and
artifact directories. That boundary is what lets the same trainer code run under
the queue, under a unit test, and from a shell one-liner.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


class Cancelled(Exception):
    """Raised inside a trainer when the operator asked the job to stop."""


@dataclass
class TrainRequest:
    run_id: int
    method: str
    base_model: str
    params: dict[str, Any]
    lora: dict[str, Any]
    out_dir: Path
    backend: str = "auto"
    train_rows: list[dict] = field(default_factory=list)
    val_rows: list[dict] = field(default_factory=list)
    # Adapter to start from — the SFT model a preference run improves.
    policy_adapter_dir: str | None = None
    reward_adapter_dir: str | None = None
    # Callable(list[str] prompts, list[str] completions) -> list[float], installed
    # by the worker so RL trainers stay ignorant of where reward comes from.
    reward_fn: Callable[[list[str], list[str]], list[float]] | None = None
    seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainResult:
    metrics: dict[str, Any]
    artifact_dir: str
    artifact_kind: str = "adapter"
    samples: list[dict] = field(default_factory=list)
    backend: str = "tiny"


class TrainContext:
    """Progress reporting, metric persistence, and cooperative cancellation."""

    def __init__(
        self,
        *,
        out_dir: Path,
        on_log: Callable[[str, str, dict], None] | None = None,
        on_progress: Callable[[dict], bool] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._on_log = on_log or (lambda level, msg, data: None)
        self._on_progress = on_progress or (lambda p: True)
        self._is_cancelled = is_cancelled or (lambda: False)
        self._metrics_path = self.out_dir / "metrics.jsonl"
        self._started = time.time()
        self._last_progress = 0.0
        self.history: list[dict] = []

    # ---------------------------------------------------------------- logging
    def log(self, message: str, level: str = "info", **data: Any) -> None:
        self._on_log(level, message, data)

    def metric(self, step: int, total: int, **values: Any) -> None:
        """Record one training step. Persists to metrics.jsonl and renews the lease.

        Throttled to one heartbeat per second. A 200-step run that writes 200
        progress rows to the database is 200 write transactions competing with
        the API for the same SQLite file, and the UI cannot render faster than
        that anyway — but every step still lands in metrics.jsonl, which is what
        the loss chart reads.
        """
        record = {"step": step, "total": total, "elapsed_s": round(time.time() - self._started, 2)}
        record.update({k: (round(v, 6) if isinstance(v, float) else v) for k, v in values.items()})
        self.history.append(record)
        with self._metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

        now = time.time()
        if now - self._last_progress >= 1.0 or step >= total:
            self._last_progress = now
            payload = {
                "step": step, "total": total,
                "pct": round(100 * step / max(total, 1), 1),
                "elapsed_s": record["elapsed_s"],
                "latest": {k: v for k, v in record.items() if k not in ("step", "total")},
            }
            if not self._on_progress(payload):
                # The lease was lost — something else owns this job now.
                raise Cancelled("lease lost; another worker owns this job")
        self.check_cancelled()

    # ------------------------------------------------------------ cancellation
    def check_cancelled(self) -> None:
        if self._is_cancelled():
            raise Cancelled("cancelled by operator")

    # --------------------------------------------------------------- artifacts
    def write_json(self, name: str, payload: Any) -> Path:
        p = self.out_dir / name
        p.write_text(json.dumps(payload, indent=2, default=str))
        return p

    def write_jsonl(self, name: str, rows: list[dict]) -> Path:
        p = self.out_dir / name
        with p.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        return p

    def summarise(self, tail: int = 20) -> dict[str, Any]:
        """Final metric block: last value and best value for every numeric series."""
        if not self.history:
            return {}
        numeric: dict[str, list[float]] = {}
        for row in self.history:
            for k, v in row.items():
                if isinstance(v, (int, float)) and k not in ("step", "total"):
                    numeric.setdefault(k, []).append(float(v))
        out: dict[str, Any] = {"steps": len(self.history),
                               "wall_seconds": round(time.time() - self._started, 1)}
        for k, vals in numeric.items():
            out[f"final_{k}"] = round(vals[-1], 6)
            recent = vals[-tail:]
            out[f"mean_{k}"] = round(sum(recent) / len(recent), 6)
            if "loss" in k:
                out[f"best_{k}"] = round(min(vals), 6)
            elif any(t in k for t in ("acc", "reward", "margin", "win")):
                out[f"best_{k}"] = round(max(vals), 6)
        return out


# ------------------------------------------------------------ backend selection

def resolve_backend(requested: str, base_model: str) -> str:
    """Decide between real training and the simulation surrogate.

    ``auto`` prefers ``torch`` — the real base model — and falls back to ``tiny``
    when the weights are not on this machine. Both run identical objective code;
    ``tiny`` swaps in a randomly-initialised micro-model (see :mod:`.tiny`). The
    fallback is *loud*: the caller logs it and the backend is stamped on the run,
    because a loss curve from a 117k-parameter noise model must never be read as
    a result.
    """
    if requested == "tiny":
        return "tiny"

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        raise RuntimeError("torch and transformers are required; pip install -r requirements.txt")

    if requested == "torch":
        return "torch"

    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(base_model, local_files_only=True)
        return "torch"
    except Exception:
        return "tiny"
