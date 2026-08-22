"""Dataset ingest, validation, and splitting.

Uploads are normalised on the way in, not on the way out. Four upload shapes are
accepted (chat messages, prompt/completion, preference triples, benchmark rows)
and each is rewritten to one canonical form on disk, so no trainer ever has to
ask "which flavour of SFT file is this?". The cost is one rewrite at upload; the
benefit is that a schema surprise surfaces while a human is looking at the upload
screen rather than forty minutes into a training run.

Validation is row-level and non-fatal by default. A 50,000-row export with nine
malformed lines should not be rejected wholesale — the nine are reported with
their line numbers and dropped, and the count of what was dropped is stored on
the dataset so nobody later wonders why the row count is off.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any, Iterator

from .config import limits

KINDS = ("sft", "preference", "prompts", "benchmark")

KIND_HELP = {
    "sft": 'Demonstrations. Either {"messages":[{"role","content"},…]} or {"prompt","completion"}.',
    "preference": 'Comparisons. {"prompt","chosen","rejected"} — optional "margin" in [0,1].',
    "prompts": 'Prompts only, for RL rollouts and review batches. {"prompt"} — optional "reference".',
    "benchmark": 'Held-out evaluation. {"prompt"} plus any of "reference", "choices"+"answer", "rubric".',
}

MAX_ERRORS_KEPT = 100


class RowError(Exception):
    pass


# ------------------------------------------------------------------ normalising

def _as_text(v: Any, field: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise RowError(f"'{field}' must be a non-empty string")
    return v


def _messages_to_prompt_completion(msgs: list) -> tuple[list[dict], str | None]:
    """Split a chat transcript into (context messages, target).

    Two shapes are common in the wild and both are accepted. When the transcript
    ends with an assistant turn, that turn is the target and everything before it
    is context. When it ends with the user, the target lives in a sibling field
    (``completion``/``output``/``response``) and ``None`` comes back so the caller
    reaches for it. Rejecting the second shape would fail a large fraction of real
    exports for no reason.
    """
    if not isinstance(msgs, list) or not msgs:
        raise RowError("'messages' must be a non-empty list")
    clean = []
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise RowError(f"messages[{i}] needs 'role' and 'content'")
        role = str(m["role"]).lower()
        if role not in ("system", "user", "assistant"):
            raise RowError(f"messages[{i}]: unknown role {role!r}")
        clean.append({"role": role, "content": str(m["content"])})
    if clean[-1]["role"] == "assistant":
        return clean[:-1], clean[-1]["content"]
    return clean, None


def normalise(kind: str, row: dict) -> dict:
    """One raw row → the canonical form stored on disk.

    Canonical forms:
        sft         {"messages": [...], "completion": str}
        preference  {"messages": [...], "chosen": str, "rejected": str, "margin": float}
        prompts     {"messages": [...], "meta": {...}}
        benchmark   {"messages": [...], "reference": str|None, "choices": [...]|None,
                     "answer": str|None, "rubric": str|None}
    """
    if not isinstance(row, dict):
        raise RowError("row is not a JSON object")

    def context() -> tuple[list[dict], str | None]:
        if "messages" in row:
            return _messages_to_prompt_completion(row["messages"])
        prompt = _as_text(row.get("prompt") or row.get("input") or row.get("instruction"),
                          "prompt")
        msgs = []
        if isinstance(row.get("system"), str) and row["system"].strip():
            msgs.append({"role": "system", "content": row["system"]})
        msgs.append({"role": "user", "content": prompt})
        return msgs, None

    if kind == "sft":
        msgs, target = context()
        if target is None:
            target = row.get("completion") or row.get("output") or row.get("response")
            if not isinstance(target, str) or not target.strip():
                raise RowError(
                    "no target to learn from: end 'messages' with an assistant turn, or "
                    "supply 'completion'")
        return {"messages": msgs, "completion": target}

    if kind == "preference":
        msgs, _ = context()
        chosen = _as_text(row.get("chosen") or row.get("preferred"), "chosen")
        rejected = _as_text(row.get("rejected") or row.get("dispreferred"), "rejected")
        if chosen.strip() == rejected.strip():
            raise RowError("'chosen' and 'rejected' are identical — no signal in this pair")
        margin = row.get("margin", 1.0)
        try:
            margin = min(1.0, max(0.0, float(margin)))
        except (TypeError, ValueError):
            raise RowError("'margin' must be a number in [0,1]")
        return {"messages": msgs, "chosen": chosen, "rejected": rejected, "margin": margin}

    if kind == "prompts":
        msgs, _ = context()
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        return {"messages": msgs, "meta": meta}

    if kind == "benchmark":
        msgs, target = context()
        reference = row.get("reference") or row.get("answer_text") or target
        choices = row.get("choices")
        if choices is not None and (not isinstance(choices, list) or len(choices) < 2):
            raise RowError("'choices' must be a list of at least two options")
        answer = row.get("answer")
        if choices and answer is None:
            raise RowError("'choices' given without 'answer'")
        if choices and str(answer) not in {str(c) for c in choices} and \
                not (isinstance(answer, int) and 0 <= answer < len(choices)):
            raise RowError("'answer' must be one of 'choices' (or its index)")
        return {
            "messages": msgs,
            "reference": reference if isinstance(reference, str) else None,
            "choices": choices,
            "answer": None if answer is None else str(answer),
            "rubric": row.get("rubric") if isinstance(row.get("rubric"), str) else None,
        }

    raise RowError(f"unknown dataset kind {kind!r}")


# --------------------------------------------------------------------- parsing

def _iter_records(raw: bytes) -> Iterator[tuple[int, Any, str | None]]:
    """Yield ``(line_number, parsed, error)`` for JSONL or a top-level JSON array.

    People paste a JSON array about as often as they upload JSONL. Detecting it
    costs one character of lookahead and saves a support round-trip.
    """
    text = raw.decode("utf-8", errors="replace")
    stripped = text.lstrip()
    if stripped.startswith("["):
        try:
            arr = json.loads(stripped)
        except json.JSONDecodeError as e:
            yield 1, None, f"file looks like a JSON array but does not parse: {e}"
            return
        for i, item in enumerate(arr, 1):
            yield i, item, None
        return

    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            yield lineno, json.loads(line), None
        except json.JSONDecodeError as e:
            yield lineno, None, f"invalid JSON: {e.msg}"


# --------------------------------------------------------------------- ingest

def ingest(raw: bytes, kind: str, dest: Path) -> dict[str, Any]:
    """Validate, normalise, and write. Returns a report for the Dataset row."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    max_rows = limits()["max_rows"]

    dest.parent.mkdir(parents=True, exist_ok=True)
    errors: list[dict] = []
    kept = 0
    bad = 0
    prompt_lens: list[int] = []
    target_lens: list[int] = []
    seen: set[str] = set()
    duplicates = 0
    digest = hashlib.sha256()

    with dest.open("w", encoding="utf-8") as fh:
        for lineno, record, parse_error in _iter_records(raw):
            if kept >= max_rows:
                errors.append({"line": lineno, "error": f"row limit {max_rows} reached; rest ignored"})
                break
            if parse_error:
                bad += 1
                if len(errors) < MAX_ERRORS_KEPT:
                    errors.append({"line": lineno, "error": parse_error})
                continue
            try:
                norm = normalise(kind, record)
            except RowError as e:
                bad += 1
                if len(errors) < MAX_ERRORS_KEPT:
                    errors.append({"line": lineno, "error": str(e)})
                continue

            key = hashlib.sha256(
                json.dumps(norm, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)

            prompt_text = "\n".join(m["content"] for m in norm["messages"])
            prompt_lens.append(len(prompt_text))
            for field in ("completion", "chosen", "reference"):
                if isinstance(norm.get(field), str):
                    target_lens.append(len(norm[field]))
                    break

            payload = json.dumps(norm, ensure_ascii=False)
            digest.update(payload.encode())
            fh.write(payload + "\n")
            kept += 1

    stats: dict[str, Any] = {
        "duplicates_dropped": duplicates,
        "prompt_chars": _describe(prompt_lens),
        "target_chars": _describe(target_lens),
    }
    if kind == "sft":
        stats["est_tokens"] = int(sum(prompt_lens + target_lens) / 4)
    if kind == "benchmark":
        stats["has_choices"] = _count_field(dest, "choices")
        stats["has_reference"] = _count_field(dest, "reference")

    return {
        "num_rows": kept,
        "num_bad_rows": bad,
        "duplicates": duplicates,
        "sha256": digest.hexdigest(),
        "bytes": dest.stat().st_size if dest.exists() else 0,
        "errors": errors,
        "stats": stats,
        "status": "ready" if kept > 0 else "invalid",
    }


def _describe(vals: list[int]) -> dict[str, float]:
    if not vals:
        return {}
    vals = sorted(vals)
    return {
        "min": vals[0], "max": vals[-1],
        "mean": round(statistics.fmean(vals), 1),
        "p50": vals[len(vals) // 2],
        "p95": vals[min(len(vals) - 1, int(len(vals) * 0.95))],
    }


def _count_field(path: Path, field: str) -> int:
    n = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if row.get(field):
                    n += 1
    return n


# --------------------------------------------------------------------- reading

def load_rows(path: str | Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def split_rows(rows: list[dict], val_fraction: float, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Deterministic hash split on prompt text.

    Hashing the prompt rather than shuffling by index means the same example lands
    on the same side of the split every time, even after the dataset grows. A
    shuffle-by-seed split silently reshuffles when rows are appended, which leaks
    yesterday's validation examples into today's training set.
    """
    if val_fraction <= 0:
        return rows, []
    train, val = [], []
    for row in rows:
        key = "\n".join(m["content"] for m in row.get("messages", []))
        h = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        (val if h < val_fraction else train).append(row)
    if not val and rows:               # tiny datasets can hash entirely one way
        val = rows[-1:]
        train = rows[:-1] or rows
    return train, val


def render_prompt(messages: list[dict]) -> str:
    """Flatten chat messages into a plain prompt string.

    Used by the simulation backend, the review console, and any tokenizer without
    a chat template. Real chat templates are applied in the trainer where the
    tokenizer is available; this is the fallback and it is intentionally boring.
    """
    parts = []
    for m in messages:
        role = m["role"]
        tag = {"system": "System", "user": "User", "assistant": "Assistant"}[role]
        parts.append(f"{tag}: {m['content']}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


_WS = re.compile(r"\s+")


def preview(path: str | Path, n: int = 5) -> list[dict]:
    out = []
    for row in load_rows(path, limit=n):
        item = {"prompt": _WS.sub(" ", render_prompt(row.get("messages", [])))[:600]}
        for field in ("completion", "chosen", "rejected", "reference", "rubric"):
            if isinstance(row.get(field), str):
                item[field] = _WS.sub(" ", row[field])[:600]
        if row.get("choices"):
            item["choices"] = row["choices"]
            item["answer"] = row.get("answer")
        out.append(item)
    return out
