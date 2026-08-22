"""The model registry: versions, promotion, and lineage.

A run produces an artifact; the registry decides what that artifact *is* to the
rest of the organisation. Three rules make that decision trustworthy:

**Promotion copies, it does not point.** ``var/runs/<id>/`` is scratch that a
retry may overwrite. Promoting snapshots the artifact into ``var/models/<id>/``,
which is never written again. A registry entry that pointed at a run directory
would silently change meaning the next time someone re-ran the job.

**One production version per name, swapped atomically.** Promoting demotes the
incumbent in the same transaction, so there is no instant where a name resolves
to two models or to none.

**Lineage is a column, not a log.** ``parent_id`` chains a DPO model to the SFT
model it improved to the base it started from, so "what produced this?" is a
query rather than an archaeology exercise.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .db import append_audit
from .models import EvalRun, ModelVersion, Run, utcnow
from .paths import model_dir


def next_version(s: Session, team_id: int, name: str) -> int:
    current = s.execute(
        select(func.max(ModelVersion.version))
        .where(ModelVersion.team_id == team_id, ModelVersion.name == name)
    ).scalar()
    return (current or 0) + 1


def register(
    s: Session, *, run: Run, artifact_dir: str, artifact_kind: str,
    metrics: dict[str, Any], name: str | None = None,
) -> ModelVersion:
    name = name or run.name
    mv = ModelVersion(
        team_id=run.team_id, name=name, version=next_version(s, run.team_id, name),
        run_id=run.id, base_model=run.base_model, parent_id=run.parent_model_version_id,
        method=run.method, artifact_kind=artifact_kind, artifact_dir=artifact_dir,
        status="draft", metrics=metrics,
    )
    s.add(mv)
    s.flush()
    append_audit(s, action="model.registered", actor_id=run.created_by, actor_type="worker",
                 team_id=run.team_id, subject_type="model_version", subject_id=mv.id,
                 payload={"run_id": run.id, "method": run.method, "version": mv.version})
    return mv


def promote(s: Session, mv: ModelVersion, *, to: str, actor_id: int) -> ModelVersion:
    """Move a version along the lifecycle. ``production`` is exclusive per name."""
    if to not in ("staging", "production", "archived"):
        raise ValueError(f"cannot promote to {to!r}")

    if to == "production":
        if mv.artifact_dir and not str(mv.artifact_dir).startswith(str(model_dir(mv.id))):
            mv.artifact_dir = str(_snapshot(mv))
        incumbents = s.execute(
            select(ModelVersion).where(
                ModelVersion.team_id == mv.team_id, ModelVersion.name == mv.name,
                ModelVersion.status == "production", ModelVersion.id != mv.id)
        ).scalars().all()
        for old in incumbents:
            old.status = "archived"
            append_audit(s, action="model.demoted", actor_id=actor_id, team_id=mv.team_id,
                         subject_type="model_version", subject_id=old.id,
                         payload={"replaced_by": mv.id})

    mv.status = to
    mv.promoted_by = actor_id
    mv.promoted_at = utcnow()
    append_audit(s, action=f"model.{to}", actor_id=actor_id, team_id=mv.team_id,
                 subject_type="model_version", subject_id=mv.id,
                 payload={"name": mv.name, "version": mv.version,
                          "metrics": mv.metrics.get("headline") if mv.metrics else None})
    return mv


def _snapshot(mv: ModelVersion) -> Path:
    dest = model_dir(mv.id)
    src = Path(mv.artifact_dir)
    if src.exists() and src.resolve() != dest.resolve():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
    return dest


def lineage(s: Session, mv: ModelVersion, limit: int = 12) -> list[dict[str, Any]]:
    """Walk parents back to the base model. Cycle-guarded — a corrupted parent
    pointer should give a short chain, not a hung request."""
    chain: list[dict[str, Any]] = []
    seen: set[int] = set()
    node: ModelVersion | None = mv
    while node is not None and node.id not in seen and len(chain) < limit:
        seen.add(node.id)
        chain.append({
            "id": node.id, "name": node.name, "version": node.version,
            "method": node.method, "status": node.status,
            "headline": (node.metrics or {}).get("headline"),
            "created_at": node.created_at,
        })
        node = s.get(ModelVersion, node.parent_id) if node.parent_id else None
    chain.append({"id": None, "name": mv.base_model, "version": None,
                  "method": "base", "status": "base", "headline": None})
    return chain


def leaderboard(s: Session, team_id: int, dataset_id: int) -> list[dict[str, Any]]:
    """Every evaluated model on one benchmark, best first.

    Restricted to a single benchmark on purpose: ranking models by scores taken on
    different datasets is the most common way a registry lies to the people
    reading it.
    """
    rows = s.execute(
        select(EvalRun, ModelVersion)
        .join(ModelVersion, ModelVersion.id == EvalRun.model_version_id)
        .where(EvalRun.team_id == team_id, EvalRun.dataset_id == dataset_id,
               EvalRun.status == "succeeded")
    ).all()

    out = []
    for ev, mv in rows:
        headline = (ev.metrics or {}).get("headline") or {}
        out.append({
            "model_version_id": mv.id, "name": mv.name, "version": mv.version,
            "method": mv.method, "status": mv.status,
            "metric": headline.get("metric"), "label": headline.get("label"),
            "value": headline.get("value"), "metrics": ev.metrics,
            "eval_id": ev.id,
        })
    return sorted(out, key=lambda r: (r["value"] is None, -(r["value"] or 0)))
