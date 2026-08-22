"""Inference serving: turn a promoted model version into something callable.

Everything upstream of this module produces artifacts. This is where an artifact
becomes a service, and the design is shaped by three facts:

**Loading a model is slow and generating is fast.** Reloading weights per request
would dominate latency entirely, so models are cached. The cache is bounded and
LRU, because an unbounded one is a memory leak with a plausible excuse — a team
with forty archived versions would otherwise pin all forty in RAM.

**A cached model is shared mutable state.** Uvicorn dispatches sync handlers onto
a threadpool, so two requests can land on the same model object concurrently.
Each cache entry carries a lock; generation holds it. That serialises inference
per model, which is the honest behaviour for a single-process server — pretending
otherwise would corrupt the KV cache and return nonsense.

**Callers should not have to know version numbers.** A service in production
should ask for ``support-bot`` and get whatever is currently promoted. Pinning
(``support-bot@3``) stays available for when a caller genuinely needs a fixed
version, which is exactly the situation where it should be explicit.
"""
from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterator

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import ModelVersion


class ModelNotFound(Exception):
    pass


# ------------------------------------------------------------------ resolution

def resolve_model(s: Session, team_id: int, spec: str) -> ModelVersion:
    """Turn a caller-supplied model string into one version row.

    Accepted forms::

        support-bot              the production version, else the newest evaluated
        support-bot@4            that exact version
        support-bot@staging      newest version with that status
        #17                      a version id
    """
    spec = (spec or "").strip()
    if not spec:
        raise ModelNotFound("no model specified")

    base = select(ModelVersion).where(
        ModelVersion.team_id == team_id,
        ModelVersion.artifact_kind == "adapter",
    )

    if spec.startswith("#"):
        mv = s.execute(base.where(ModelVersion.id == _int(spec[1:]))).scalar_one_or_none()
        if mv is None:
            raise ModelNotFound(f"no model version {spec}")
        return mv

    name, _, qualifier = spec.partition("@")
    q = base.where(ModelVersion.name == name)

    if qualifier.isdigit():
        mv = s.execute(q.where(ModelVersion.version == int(qualifier))).scalar_one_or_none()
        if mv is None:
            raise ModelNotFound(f"{name} has no version {qualifier}")
        return mv

    if qualifier:
        mv = s.execute(
            q.where(ModelVersion.status == qualifier)
             .order_by(ModelVersion.version.desc()).limit(1)
        ).scalar_one_or_none()
        if mv is None:
            raise ModelNotFound(f"{name} has no version in status '{qualifier}'")
        return mv

    for status in ("production", "staging", "evaluated", "draft"):
        mv = s.execute(
            q.where(ModelVersion.status == status)
             .order_by(ModelVersion.version.desc()).limit(1)
        ).scalar_one_or_none()
        if mv is not None:
            return mv
    raise ModelNotFound(f"no model named '{name}' in this team")


def _int(v: str) -> int:
    try:
        return int(v)
    except ValueError:
        raise ModelNotFound(f"'{v}' is not a version id")


def servable(s: Session, team_id: int) -> list[ModelVersion]:
    """Versions a caller could name. Archived ones stay reachable by explicit
    id — a service pinned to one should not break the moment it is superseded —
    but they are not advertised."""
    return list(s.execute(
        select(ModelVersion).where(
            ModelVersion.team_id == team_id,
            ModelVersion.artifact_kind == "adapter",
            ModelVersion.status.in_(("production", "staging", "evaluated")),
        ).order_by(ModelVersion.name, ModelVersion.version.desc())
    ).scalars())


# ----------------------------------------------------------------------- cache

@dataclass
class Loaded:
    policy: Any
    version_id: int
    label: str
    backend: str
    loaded_at: float
    lock: threading.Lock = field(default_factory=threading.Lock)


class ModelCache:
    def __init__(self, capacity: int | None = None):
        self.capacity = capacity or int(os.environ.get("FOUNDRY_SERVING_CACHE", "2"))
        self._entries: OrderedDict[int, Loaded] = OrderedDict()
        self._guard = threading.Lock()          # protects the dict, not generation

    def get(self, mv: ModelVersion) -> Loaded:
        with self._guard:
            hit = self._entries.get(mv.id)
            if hit is not None:
                self._entries.move_to_end(mv.id)
                return hit

        # Loaded outside the guard: it takes seconds, and holding the lock would
        # stall every other model's requests behind this one. A concurrent
        # duplicate load is possible and costs memory briefly, never correctness.
        loaded = self._load(mv)

        with self._guard:
            self._entries[mv.id] = loaded
            self._entries.move_to_end(mv.id)
            while len(self._entries) > self.capacity:
                _, victim = self._entries.popitem(last=False)
                del victim
        return loaded

    def _load(self, mv: ModelVersion) -> Loaded:
        from .trainers.base import TrainRequest, resolve_backend
        from .trainers.factory import build_policy

        backend = resolve_backend("auto", mv.base_model)
        stub = TrainRequest(
            run_id=0, method="serve", base_model=mv.base_model, params={}, lora={},
            out_dir=__import__("pathlib").Path("/tmp"),
            train_rows=[{"messages": [{"role": "user", "content": "warmup"}],
                         "completion": "warmup"}],
            val_rows=[],
        )
        policy = build_policy(stub, backend, adapter_dir=mv.artifact_dir, inference_only=True)
        return Loaded(policy=policy, version_id=mv.id,
                      label=f"{mv.name}@{mv.version}", backend=backend,
                      loaded_at=time.time())

    def evict(self, version_id: int) -> bool:
        with self._guard:
            return self._entries.pop(version_id, None) is not None

    def stats(self) -> dict[str, Any]:
        with self._guard:
            return {
                "capacity": self.capacity,
                "loaded": [
                    {"model_version_id": k, "label": v.label, "backend": v.backend,
                     "age_seconds": round(time.time() - v.loaded_at, 1)}
                    for k, v in self._entries.items()
                ],
            }


CACHE = ModelCache()


# ------------------------------------------------------------------ generation

def _prompt_for(loaded: Loaded, messages: list[dict]) -> str:
    from .trainers.policy import render
    return render(loaded.policy.tokenizer, messages)


def count_tokens(loaded: Loaded, text: str) -> int:
    return len(loaded.policy.tokenizer(text, add_special_tokens=False)["input_ids"])


def complete(loaded: Loaded, messages: list[dict], *, max_tokens: int = 256,
             temperature: float = 0.7, top_p: float = 1.0, n: int = 1) -> dict[str, Any]:
    """One blocking completion. Returns text plus a usage block."""
    from .trainers.policy import generate

    prompt = _prompt_for(loaded, messages)
    with loaded.lock:
        outs = generate(loaded.policy, [prompt], max_new_tokens=max_tokens,
                        temperature=temperature, top_p=top_p, num_return_sequences=n)[0]

    prompt_tokens = count_tokens(loaded, prompt)
    completion_tokens = sum(count_tokens(loaded, o) for o in outs)
    return {
        "choices": outs,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def stream(loaded: Loaded, messages: list[dict], *, max_tokens: int = 256,
           temperature: float = 0.7, top_p: float = 1.0) -> Iterator[str]:
    """Yield text fragments as they are produced.

    ``generate`` blocks, so it runs on a worker thread and hands tokens back
    through a streamer queue. The lock is held for the whole generation — the
    model cannot serve anyone else mid-sequence without corrupting its KV cache.
    """
    import torch
    from transformers import TextIteratorStreamer

    tok = loaded.policy.tokenizer
    prompt = _prompt_for(loaded, messages)

    with loaded.lock:
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        old_side, tok.padding_side = tok.padding_side, "left"
        try:
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1024,
                      add_special_tokens=False).to(loaded.policy.device)
            kwargs = dict(
                **enc, max_new_tokens=max_tokens, do_sample=temperature > 0,
                temperature=max(temperature, 1e-5), top_p=top_p,
                pad_token_id=tok.pad_token_id, streamer=streamer,
            )
            thread = threading.Thread(target=_generate_quietly,
                                      args=(loaded.policy.model, kwargs), daemon=True)
            thread.start()
            for piece in streamer:
                if piece:
                    yield piece
            thread.join(timeout=5)
        finally:
            tok.padding_side = old_side


def _generate_quietly(model, kwargs) -> None:
    """Generation on a background thread.

    A raised exception here would vanish into the thread and leave the consumer
    blocked on an iterator that never ends; the streamer is closed instead so the
    caller sees the stream finish rather than hanging.
    """
    import torch
    try:
        with torch.no_grad():
            model.generate(**kwargs)
    except Exception:
        streamer = kwargs.get("streamer")
        if streamer is not None:
            try:
                streamer.end()
            except Exception:
                pass
