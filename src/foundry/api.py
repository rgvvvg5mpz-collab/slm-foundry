"""FastAPI surface for SLM Foundry.

Two invariants hold across every handler:

**Nothing slow runs here.** Training, generation, judging and evaluation are all
enqueued and executed by workers. The one exception is the playground, which is
an explicit interactive request for a single short generation, and it says so.

**Every query is scoped to the caller's team.** ``_owned`` is the single accessor
for fetching a row by id, and it filters on ``team_id`` before returning it.
There is no code path that fetches by primary key alone, which is what stops a
tenancy bug from being one forgotten ``.where()`` away.

Uploads take the file as the raw request body rather than multipart form data.
That drops a dependency, streams cleanly, and is one line on the browser side
(``fetch(url, {method: 'POST', body: file})``).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import auth as authlib
from . import datasets as dslib
from . import queue as q
from . import registry as reglib
from . import review as reviewlib
from .config import (
    base_models, cfg, defaults_for, judge_config, limits, method_names,
    method_spec, methods, validate_params,
)
from .db import append_audit, get_session, init_db, session_factory, verify_audit_chain
from .models import (
    Adjudication, Annotation, Dataset, EvalRun, Job, JobEvent, ModelVersion,
    PreferencePair, ReviewAssignment, ReviewBatch, ReviewItem, Run, Team, User,
    utcnow,
)
from .paths import static_dir, upload_dir
from .schemas import (
    AdjudicationRequest, AnnotationRequest, DatasetCreate, EvaluateRequest,
    LoginRequest, LoginResponse, PlaygroundRequest, PromoteRequest,
    ReviewBatchCreate, RunCreate, TeamCreate, UserCreate, UserOut,
)

app = FastAPI(title="SLM Foundry", version="1.0.0",
              description="Bring your data, get a fine-tuned small language model.")


# ---------------------------------------------------------------- dependencies

def db() -> Session:
    s = session_factory()()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def current_user(authorization: str | None = Header(None), s: Session = Depends(db)) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "missing bearer token")
    user = authlib.resolve_token(s, authorization.split(None, 1)[1])
    if user is None:
        raise HTTPException(401, "invalid or expired token")
    return user


def _require(minimum: str):
    def dep(user: User = Depends(current_user)) -> User:
        if not authlib.has_role(user, minimum):
            raise HTTPException(403, f"{minimum} role required")
        return user
    return dep


require_member = _require("member")
require_reviewer = _require("reviewer")
require_lead = _require("lead")
require_admin = _require("admin")


def _owned(s: Session, model, obj_id: int | None, user: User, what: str):
    """Fetch by id *and* team. The only sanctioned way to load a tenant row."""
    if obj_id is None:
        return None
    obj = s.get(model, obj_id)
    if obj is None or obj.team_id != user.team_id:
        raise HTTPException(404, f"{what} {obj_id} not found")
    return obj


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.exception_handler(ValueError)
def _value_error(_request: Request, exc: ValueError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


# ------------------------------------------------------------------------ auth

@app.post("/api/login", response_model=LoginResponse)
def api_login(req: LoginRequest, s: Session = Depends(db)):
    result = authlib.login(s, req.email, req.password)
    if result is None:
        raise HTTPException(401, "invalid credentials")
    user, token = result
    return LoginResponse(token=token, user=_user_out(user))


@app.post("/api/logout")
def api_logout(authorization: str = Header(...), s: Session = Depends(db)):
    authlib.revoke_token(s, authorization.split(None, 1)[1])
    return {"ok": True}


@app.get("/api/me", response_model=UserOut)
def api_me(user: User = Depends(current_user)):
    return _user_out(user)


def _user_out(user: User) -> UserOut:
    return UserOut(id=user.id, email=user.email, name=user.name, role=user.role,
                   team_id=user.team_id, team_name=user.team.name, team_slug=user.team.slug)


# --------------------------------------------------------------------- catalog

@app.get("/api/catalog")
def api_catalog(_user: User = Depends(current_user)):
    """Everything the run wizard needs to draw itself. One request, cached client-side."""
    return {
        "methods": methods(),
        "method_order": ["sft", "dpo", "grpo", "gspo", "ppo", "rlaif", "reward"],
        "base_models": base_models(),
        "dataset_kinds": [{"kind": k, "help": h} for k, h in dslib.KIND_HELP.items()],
        "judge": judge_config(),
        "limits": limits(),
    }


# -------------------------------------------------------------------- datasets

@app.get("/api/datasets")
def api_datasets(kind: str | None = None, user: User = Depends(current_user),
                 s: Session = Depends(db)):
    query = select(Dataset).where(Dataset.team_id == user.team_id)
    if kind:
        query = query.where(Dataset.kind == kind)
    rows = s.execute(query.order_by(Dataset.created_at.desc())).scalars().all()
    return [_dataset_out(d) for d in rows]


def _dataset_out(d: Dataset) -> dict[str, Any]:
    return {
        "id": d.id, "name": d.name, "version": d.version, "kind": d.kind,
        "description": d.description, "status": d.status, "num_rows": d.num_rows,
        "num_bad_rows": d.num_bad_rows, "bytes": d.bytes, "sha256": d.sha256[:12],
        "stats": d.stats, "errors": (d.errors or [])[:20],
        "created_at": d.created_at,
    }


@app.post("/api/datasets/upload")
async def api_upload(
    request: Request,
    name: str = Query(...),
    kind: str = Query(...),
    description: str = Query(""),
    user: User = Depends(require_member),
):
    """Upload a JSONL (or JSON array) file as the raw request body."""
    meta = DatasetCreate(name=name, kind=kind, description=description)
    raw = await request.body()
    if not raw:
        raise HTTPException(400, "empty upload")
    if len(raw) > limits()["max_upload_bytes"]:
        raise HTTPException(413, f"file exceeds {limits()['max_upload_bytes']} bytes")

    with get_session() as s:
        version = (s.execute(
            select(func.max(Dataset.version)).where(
                Dataset.team_id == user.team_id, Dataset.name == meta.name)
        ).scalar() or 0) + 1
        ds = Dataset(team_id=user.team_id, name=meta.name, version=version, kind=meta.kind,
                     description=meta.description, status="validating", created_by=user.id)
        s.add(ds)
        s.flush()
        dataset_id = ds.id
        team_slug = user.team.slug

    dest = upload_dir(team_slug, dataset_id) / "data.jsonl"
    report = dslib.ingest(raw, meta.kind, dest)

    with get_session() as s:
        ds = s.get(Dataset, dataset_id)
        ds.path = str(dest)
        ds.status = report["status"]
        ds.num_rows = report["num_rows"]
        ds.num_bad_rows = report["num_bad_rows"]
        ds.bytes = report["bytes"]
        ds.sha256 = report["sha256"]
        ds.stats = report["stats"]
        ds.errors = report["errors"]
        append_audit(s, action="dataset.uploaded", actor_id=user.id, team_id=user.team_id,
                     subject_type="dataset", subject_id=ds.id,
                     payload={"kind": ds.kind, "rows": ds.num_rows, "sha256": ds.sha256})
        out = _dataset_out(ds)
    return out


@app.get("/api/datasets/{dataset_id}")
def api_dataset(dataset_id: int, user: User = Depends(current_user), s: Session = Depends(db)):
    ds = _owned(s, Dataset, dataset_id, user, "dataset")
    return {**_dataset_out(ds), "preview": dslib.preview(ds.path, 8) if ds.path else []}


@app.delete("/api/datasets/{dataset_id}")
def api_delete_dataset(dataset_id: int, user: User = Depends(require_member),
                       s: Session = Depends(db)):
    ds = _owned(s, Dataset, dataset_id, user, "dataset")
    used = s.execute(
        select(func.count()).select_from(Run).where(
            (Run.train_dataset_id == dataset_id) | (Run.eval_dataset_id == dataset_id))
    ).scalar()
    if used:
        # A run's lineage points here. Deleting the dataset would leave every run
        # that used it claiming a provenance nobody can check.
        raise HTTPException(409, f"{used} run(s) reference this dataset; archive it instead")
    s.delete(ds)
    append_audit(s, action="dataset.deleted", actor_id=user.id, team_id=user.team_id,
                 subject_type="dataset", subject_id=dataset_id, payload={"name": ds.name})
    return {"ok": True}


# ------------------------------------------------------------------------ runs

@app.get("/api/runs")
def api_runs(status: str | None = None, limit: int = 50,
             user: User = Depends(current_user), s: Session = Depends(db)):
    query = select(Run).where(Run.team_id == user.team_id)
    if status:
        query = query.where(Run.status == status)
    runs = s.execute(query.order_by(Run.id.desc()).limit(limit)).scalars().all()
    jobs = {j.run_id: j for j in s.execute(
        select(Job).where(Job.run_id.in_([r.id for r in runs] or [0]), Job.kind == "train")
    ).scalars()}
    return [_run_out(r, jobs.get(r.id)) for r in runs]


def _run_out(r: Run, job: Job | None = None) -> dict[str, Any]:
    return {
        "id": r.id, "name": r.name, "method": r.method, "base_model": r.base_model,
        "status": r.status, "backend": r.backend, "metrics": r.metrics or {},
        "params": r.params, "lora": r.lora, "error": r.error,
        "train_dataset_id": r.train_dataset_id, "eval_dataset_id": r.eval_dataset_id,
        "review_batch_id": r.review_batch_id,
        "parent_model_version_id": r.parent_model_version_id,
        "created_at": r.created_at, "started_at": r.started_at, "finished_at": r.finished_at,
        "job": {"id": job.id, "status": job.status, "progress": job.progress,
                "attempts": job.attempts, "priority": job.priority} if job else None,
    }


@app.post("/api/runs")
def api_create_run(req: RunCreate, user: User = Depends(require_member),
                   s: Session = Depends(db)):
    if req.method not in method_names():
        raise HTTPException(400, f"unknown method {req.method!r}")
    spec = method_spec(req.method)
    params, errors = validate_params(req.method, req.params)

    needs = spec.get("requires", {})
    train_kind = needs.get("train")
    train_ds = _owned(s, Dataset, req.train_dataset_id, user, "dataset")
    if train_ds and train_kind and train_ds.kind != train_kind:
        raise HTTPException(400,
                            f"{spec['label']} needs a '{train_kind}' dataset, "
                            f"but dataset {train_ds.id} is '{train_ds.kind}'")
    if train_ds and train_ds.status != "ready":
        raise HTTPException(400, f"dataset {train_ds.id} is {train_ds.status}, not ready")

    if not train_ds:
        if req.method in ("dpo", "reward"):
            available = s.execute(
                select(func.count()).select_from(PreferencePair)
                .where(PreferencePair.team_id == user.team_id)
            ).scalar()
            if not available:
                raise HTTPException(400,
                                    "no preference data: upload a preference dataset or "
                                    "close a review batch first")
        else:
            raise HTTPException(400, f"{spec['label']} needs a {train_kind} dataset")

    eval_ds = _owned(s, Dataset, req.eval_dataset_id, user, "dataset")
    if eval_ds and eval_ds.kind != "benchmark":
        raise HTTPException(400, "the evaluation dataset must be of kind 'benchmark'")

    parent = _owned(s, ModelVersion, req.parent_model_version_id, user, "model version")
    reward_mv = _owned(s, ModelVersion, req.reward_model_version_id, user, "model version")
    if needs.get("reward_model") and reward_mv is None:
        raise HTTPException(400, "PPO needs a reward model — train one, or choose GRPO/GSPO "
                                 "which derive their baseline from the sample group")
    if reward_mv and reward_mv.artifact_kind != "reward":
        raise HTTPException(400, f"model version {reward_mv.id} is not a reward model")
    if needs.get("policy") and parent is None:
        # Not fatal: preference methods on a raw base model are legitimate, just
        # rarely what someone means to do.
        pass

    _owned(s, ReviewBatch, req.review_batch_id, user, "review batch")

    run = Run(
        team_id=user.team_id, name=req.name, method=req.method, base_model=req.base_model,
        parent_model_version_id=req.parent_model_version_id,
        reward_model_version_id=req.reward_model_version_id,
        train_dataset_id=req.train_dataset_id, eval_dataset_id=req.eval_dataset_id,
        review_batch_id=req.review_batch_id,
        params=params, lora=req.lora.model_dump(), backend=req.backend,
        status="queued", created_by=user.id,
    )
    s.add(run)
    s.flush()

    job = q.enqueue(s, team_id=user.team_id, kind="train", payload={"run_id": run.id},
                    run_id=run.id, priority=req.priority, created_by=user.id)
    append_audit(s, action="run.submitted", actor_id=user.id, team_id=user.team_id,
                 subject_type="run", subject_id=run.id,
                 payload={"method": req.method, "base_model": req.base_model,
                          "params": params})
    return {**_run_out(run, job), "warnings": errors}


@app.get("/api/runs/{run_id}")
def api_run(run_id: int, user: User = Depends(current_user), s: Session = Depends(db)):
    run = _owned(s, Run, run_id, user, "run")
    job = s.execute(
        select(Job).where(Job.run_id == run_id, Job.kind == "train")
        .order_by(Job.id.desc()).limit(1)
    ).scalar_one_or_none()
    mv = s.execute(
        select(ModelVersion).where(ModelVersion.run_id == run_id)
    ).scalar_one_or_none()
    return {
        **_run_out(run, job),
        "model_version": _model_out(mv) if mv else None,
        "series": _metrics_series(run),
        "samples": _run_samples(run),
    }


def _metrics_series(run: Run) -> list[dict]:
    """The loss chart's data, straight off disk.

    Read from ``metrics.jsonl`` rather than a table: it is written once per step
    by a process that must not be contending with the API for a write lock, and
    it is exactly the file an operator would `tail`.
    """
    import json
    path = Path(run.artifact_dir).parent / "metrics.jsonl" if run.artifact_dir else None
    if path is None or not path.exists():
        from .paths import run_dir
        path = run_dir(run.id) / "metrics.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if len(rows) > 400:                       # downsample for the browser
        stride = len(rows) // 400 + 1
        rows = rows[::stride] + rows[-1:]
    return rows


def _run_samples(run: Run) -> list[dict]:
    import json
    from .paths import run_dir
    path = run_dir(run.id) / "samples.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()][:12]


@app.get("/api/runs/{run_id}/events")
def api_run_events(run_id: int, after: int = 0, user: User = Depends(current_user),
                   s: Session = Depends(db)):
    run = _owned(s, Run, run_id, user, "run")
    job_ids = list(s.execute(select(Job.id).where(Job.run_id == run.id)).scalars())
    if not job_ids:
        return {"events": [], "cursor": after}
    events = s.execute(
        select(JobEvent).where(JobEvent.job_id.in_(job_ids), JobEvent.id > after)
        .order_by(JobEvent.id.asc()).limit(300)
    ).scalars().all()
    return {
        "events": [{"id": e.id, "ts": e.ts, "level": e.level, "message": e.message,
                    "data": e.data} for e in events],
        "cursor": events[-1].id if events else after,
    }


@app.post("/api/runs/{run_id}/cancel")
def api_cancel_run(run_id: int, user: User = Depends(require_member), s: Session = Depends(db)):
    run = _owned(s, Run, run_id, user, "run")
    jobs = s.execute(select(Job).where(Job.run_id == run.id,
                                       Job.status.in_(("queued", "running")))).scalars().all()
    for job in jobs:
        q.request_cancel(s, job.id)
    if not jobs and run.status in ("queued", "running"):
        run.status = "cancelled"
    append_audit(s, action="run.cancelled", actor_id=user.id, team_id=user.team_id,
                 subject_type="run", subject_id=run.id, payload={"jobs": [j.id for j in jobs]})
    return {"ok": True, "cancelled_jobs": [j.id for j in jobs]}


# ----------------------------------------------------------------------- queue

@app.get("/api/queue")
def api_queue(user: User = Depends(current_user), s: Session = Depends(db)):
    """Cluster-wide queue health, plus this team's own jobs.

    Other teams' job *counts* are visible and their contents are not. Seeing that
    the queue is forty deep is what stops someone from concluding the service is
    broken and resubmitting four more times.
    """
    stats = q.queue_stats()
    mine = s.execute(
        select(Job).where(Job.team_id == user.team_id,
                          Job.status.in_(("queued", "running")))
        .order_by(Job.priority.desc(), Job.id.asc())
    ).scalars().all()
    recent = s.execute(
        select(Job).where(Job.team_id == user.team_id,
                          Job.status.in_(("succeeded", "failed", "cancelled")))
        .order_by(Job.id.desc()).limit(15)
    ).scalars().all()
    teams = {t.id: t for t in s.execute(select(Team)).scalars()}
    my_team = teams.get(user.team_id)

    ahead = s.execute(
        select(func.count()).select_from(Job).where(Job.status == "queued")
    ).scalar()

    return {
        "workers": stats["workers"],
        "by_status": stats["by_status"],
        "cluster_queued": ahead,
        "team": {"name": my_team.name if my_team else "", "slug": my_team.slug if my_team else "",
                 "concurrency": my_team.concurrency if my_team else 0,
                 "running": sum(1 for j in mine if j.status == "running")},
        "fair_share": [
            {"team": teams[row["team_id"]].name if row["team_id"] in teams else "?",
             "status": row["status"], "count": row["count"]}
            for row in stats["by_team"]
        ],
        "active": [_job_out(j) for j in mine],
        "recent": [_job_out(j) for j in recent],
    }


def _job_out(j: Job) -> dict[str, Any]:
    return {
        "id": j.id, "kind": j.kind, "run_id": j.run_id, "status": j.status,
        "priority": j.priority, "attempts": j.attempts, "max_attempts": j.max_attempts,
        "progress": j.progress or {}, "worker_id": j.worker_id, "error": j.error[:400],
        "created_at": j.created_at, "started_at": j.started_at, "finished_at": j.finished_at,
    }


@app.get("/api/jobs/{job_id}/events")
def api_job_events(job_id: int, after: int = 0, user: User = Depends(current_user),
                   s: Session = Depends(db)):
    job = _owned(s, Job, job_id, user, "job")
    events = s.execute(
        select(JobEvent).where(JobEvent.job_id == job.id, JobEvent.id > after)
        .order_by(JobEvent.id.asc()).limit(300)
    ).scalars().all()
    return {"events": [{"id": e.id, "ts": e.ts, "level": e.level, "message": e.message,
                        "data": e.data} for e in events],
            "cursor": events[-1].id if events else after}


@app.post("/api/jobs/{job_id}/cancel")
def api_cancel_job(job_id: int, user: User = Depends(require_member), s: Session = Depends(db)):
    job = _owned(s, Job, job_id, user, "job")
    return {"ok": q.request_cancel(s, job.id)}


# ----------------------------------------------------------------- model registry

@app.get("/api/models")
def api_models(status: str | None = None, kind: str | None = None,
               user: User = Depends(current_user), s: Session = Depends(db)):
    query = select(ModelVersion).where(ModelVersion.team_id == user.team_id)
    if status:
        query = query.where(ModelVersion.status == status)
    if kind:
        query = query.where(ModelVersion.artifact_kind == kind)
    rows = s.execute(query.order_by(ModelVersion.id.desc())).scalars().all()
    return [_model_out(m) for m in rows]


def _model_out(m: ModelVersion) -> dict[str, Any]:
    metrics = m.metrics or {}
    return {
        "id": m.id, "name": m.name, "version": m.version, "method": m.method,
        "base_model": m.base_model, "status": m.status, "artifact_kind": m.artifact_kind,
        "run_id": m.run_id, "parent_id": m.parent_id,
        "headline": metrics.get("headline"), "metrics": metrics,
        "backend": metrics.get("backend"), "notes": m.notes,
        "created_at": m.created_at, "promoted_at": m.promoted_at,
    }


@app.get("/api/models/{model_id}")
def api_model(model_id: int, user: User = Depends(current_user), s: Session = Depends(db)):
    mv = _owned(s, ModelVersion, model_id, user, "model version")
    evals = s.execute(
        select(EvalRun, Dataset).join(Dataset, Dataset.id == EvalRun.dataset_id)
        .where(EvalRun.model_version_id == mv.id)
    ).all()
    return {
        **_model_out(mv),
        "lineage": reglib.lineage(s, mv),
        "evals": [{"id": e.id, "dataset": d.name, "dataset_id": d.id, "status": e.status,
                   "metrics": e.metrics} for e, d in evals],
    }


@app.post("/api/models/{model_id}/promote")
def api_promote(model_id: int, req: PromoteRequest, user: User = Depends(require_lead),
                s: Session = Depends(db)):
    mv = _owned(s, ModelVersion, model_id, user, "model version")
    if req.to == "production" and mv.status == "draft":
        raise HTTPException(400,
                            "this version has not been benchmarked. Run an evaluation "
                            "before promoting it to production.")
    if req.notes:
        mv.notes = req.notes
    reglib.promote(s, mv, to=req.to, actor_id=user.id)
    return _model_out(mv)


@app.post("/api/models/{model_id}/evaluate")
def api_evaluate(model_id: int, req: EvaluateRequest, user: User = Depends(require_member),
                 s: Session = Depends(db)):
    mv = _owned(s, ModelVersion, model_id, user, "model version")
    ds = _owned(s, Dataset, req.dataset_id, user, "dataset")
    if ds.kind != "benchmark":
        raise HTTPException(400, "evaluation needs a 'benchmark' dataset")

    existing = s.execute(
        select(EvalRun).where(EvalRun.model_version_id == mv.id, EvalRun.dataset_id == ds.id)
    ).scalar_one_or_none()
    if existing and existing.status in ("queued", "running"):
        return {"eval_run_id": existing.id, "status": existing.status}
    if existing:
        s.delete(existing)
        s.flush()

    ev = EvalRun(team_id=user.team_id, model_version_id=mv.id, dataset_id=ds.id,
                 status="queued")
    s.add(ev)
    s.flush()
    job = q.enqueue(s, team_id=user.team_id, kind="eval", payload={"eval_run_id": ev.id},
                    priority=5, created_by=user.id)
    return {"eval_run_id": ev.id, "job_id": job.id, "status": "queued"}


@app.get("/api/leaderboard")
def api_leaderboard(dataset_id: int, user: User = Depends(current_user),
                    s: Session = Depends(db)):
    _owned(s, Dataset, dataset_id, user, "dataset")
    return reglib.leaderboard(s, user.team_id, dataset_id)


# ---------------------------------------------------------------------- review

@app.get("/api/review/batches")
def api_batches(user: User = Depends(current_user), s: Session = Depends(db)):
    batches = s.execute(
        select(ReviewBatch).where(ReviewBatch.team_id == user.team_id)
        .order_by(ReviewBatch.id.desc())
    ).scalars().all()
    counts = dict(s.execute(
        select(ReviewItem.batch_id, func.count()).where(
            ReviewItem.team_id == user.team_id).group_by(ReviewItem.batch_id)
    ).all())
    done = dict(s.execute(
        select(ReviewItem.batch_id, func.count()).where(
            ReviewItem.team_id == user.team_id,
            ReviewItem.status.in_(("complete", "skipped"))).group_by(ReviewItem.batch_id)
    ).all())
    pairs = dict(s.execute(
        select(PreferencePair.batch_id, func.count()).where(
            PreferencePair.team_id == user.team_id).group_by(PreferencePair.batch_id)
    ).all())
    return [{
        "id": b.id, "name": b.name, "protocol": b.protocol, "status": b.status,
        "candidates_per_prompt": b.candidates_per_prompt,
        "annotations_per_item": b.annotations_per_item,
        "ai_assist_fraction": b.ai_assist_fraction, "judge_model": b.judge_model,
        "items": counts.get(b.id, 0), "reviewed": done.get(b.id, 0),
        "pairs": pairs.get(b.id, 0), "created_at": b.created_at,
    } for b in batches]


@app.post("/api/review/batches")
def api_create_batch(req: ReviewBatchCreate, user: User = Depends(require_member),
                     s: Session = Depends(db)):
    ds = _owned(s, Dataset, req.prompt_dataset_id, user, "dataset")
    if ds.kind != "prompts":
        raise HTTPException(400, "a review batch samples from a 'prompts' dataset")
    _owned(s, ModelVersion, req.policy_model_version_id, user, "model version")

    batch = ReviewBatch(
        team_id=user.team_id, name=req.name, protocol=req.protocol,
        policy_model_version_id=req.policy_model_version_id,
        prompt_dataset_id=req.prompt_dataset_id,
        candidates_per_prompt=req.candidates_per_prompt,
        annotations_per_item=req.annotations_per_item,
        ai_assist_fraction=req.ai_assist_fraction,
        judge_model=req.judge_model or judge_config()["model"],
        status="draft", created_by=user.id,
    )
    s.add(batch)
    s.flush()
    job = q.enqueue(s, team_id=user.team_id, kind="generate",
                    payload={"batch_id": batch.id, "limit": req.limit},
                    priority=6, created_by=user.id)
    append_audit(s, action="review.batch_created", actor_id=user.id, team_id=user.team_id,
                 subject_type="review_batch", subject_id=batch.id,
                 payload={"protocol": req.protocol, "limit": req.limit})
    return {"id": batch.id, "job_id": job.id, "status": batch.status}


@app.post("/api/review/batches/{batch_id}/assemble")
def api_assemble(batch_id: int, user: User = Depends(require_member), s: Session = Depends(db)):
    batch = _owned(s, ReviewBatch, batch_id, user, "review batch")
    job = q.enqueue(s, team_id=user.team_id, kind="assemble",
                    payload={"batch_id": batch.id}, priority=7, created_by=user.id)
    return {"job_id": job.id}


@app.post("/api/review/batches/{batch_id}/close")
def api_close_batch(batch_id: int, user: User = Depends(require_member),
                    s: Session = Depends(db)):
    batch = _owned(s, ReviewBatch, batch_id, user, "review batch")
    batch.status = "closed"
    return {"ok": True}


@app.get("/api/review/next")
def api_review_next(batch_id: int | None = None, user: User = Depends(require_reviewer),
                    s: Session = Depends(db)):
    result = reviewlib.assign_next(s, user, batch_id)
    if result is None:
        remaining = s.execute(
            select(func.count()).select_from(ReviewItem)
            .where(ReviewItem.team_id == user.team_id, ReviewItem.status == "pending")
        ).scalar()
        return {"item": None, "queue_remaining": remaining}

    item, prob = result
    batch = s.get(ReviewBatch, item.batch_id)
    remaining = s.execute(
        select(func.count()).select_from(ReviewItem)
        .where(ReviewItem.batch_id == item.batch_id,
               ReviewItem.status.in_(("pending", "in_review")))
    ).scalar()

    ai = s.execute(
        select(Annotation).where(Annotation.item_id == item.id,
                                 Annotation.annotator_type == "ai",
                                 Annotation.superseded_by.is_(None))
    ).scalar_one_or_none()

    return {
        "item": {
            "id": item.id, "batch_id": item.batch_id, "batch_name": batch.name,
            "protocol": batch.protocol, "prompt": item.prompt,
            "candidates": reviewlib.presentation_order(item, user.id),
            "uncertainty": item.uncertainty, "meta": item.meta,
            "sampling_prob": round(prob, 5),
            "rubric": judge_config()["rubric"] if batch.protocol == "rubric" else None,
            # Shown behind a disclosure, never as a default. A reviewer who reads
            # the judge's answer first is confirming it, not judging.
            "ai_suggestion": {"choice": ai.choice, "rationale": ai.rationale,
                              "model": ai.judge_model} if ai else None,
        },
        "queue_remaining": remaining,
    }


@app.post("/api/review/annotate")
def api_annotate(req: AnnotationRequest, user: User = Depends(require_reviewer),
                 s: Session = Depends(db)):
    item = _owned(s, ReviewItem, req.item_id, user, "review item")
    batch = s.get(ReviewBatch, item.batch_id)
    valid = {c["id"] for c in (item.candidates or [])} | set(reviewlib.TIE_CHOICES)
    if batch.protocol == "pairwise" and req.choice not in valid:
        raise HTTPException(400, f"choice must be one of {sorted(valid)}")

    reviewlib.record_annotation(
        s, item=item, user=user, protocol=batch.protocol, choice=req.choice,
        ranking=req.ranking, rubric_scores=req.rubric_scores, rationale=req.rationale,
        confidence=req.confidence, latency_ms=req.latency_ms,
    )
    return {"ok": True, "item_status": item.status}


@app.post("/api/review/skip")
def api_skip(req: AnnotationRequest, user: User = Depends(require_reviewer),
             s: Session = Depends(db)):
    item = _owned(s, ReviewItem, req.item_id, user, "review item")
    assignment = s.execute(
        select(ReviewAssignment).where(ReviewAssignment.item_id == item.id,
                                       ReviewAssignment.user_id == user.id)
    ).scalar_one_or_none()
    if assignment:
        assignment.status = "skipped"
        assignment.completed_at = utcnow()
    return {"ok": True}


@app.get("/api/review/disputed")
def api_disputed(user: User = Depends(require_lead), s: Session = Depends(db)):
    items = s.execute(
        select(ReviewItem).where(ReviewItem.team_id == user.team_id,
                                 ReviewItem.status == "disputed").limit(50)
    ).scalars().all()
    out = []
    for item in items:
        anns = s.execute(
            select(Annotation, User).outerjoin(User, User.id == Annotation.user_id)
            .where(Annotation.item_id == item.id, Annotation.superseded_by.is_(None))
        ).all()
        out.append({
            "id": item.id, "prompt": item.prompt, "candidates": item.candidates,
            "annotations": [{
                "who": (u.name if u else f"AI · {a.judge_model}"),
                "type": a.annotator_type, "choice": a.choice,
                "confidence": a.confidence, "rationale": a.rationale,
            } for a, u in anns],
        })
    return out


@app.post("/api/review/adjudicate")
def api_adjudicate(req: AdjudicationRequest, user: User = Depends(require_lead),
                   s: Session = Depends(db)):
    item = _owned(s, ReviewItem, req.item_id, user, "review item")
    existing = s.execute(
        select(Adjudication).where(Adjudication.item_id == item.id)
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(409, "already adjudicated")
    s.add(Adjudication(item_id=item.id, team_id=item.team_id, user_id=user.id,
                       choice=req.choice, rationale=req.rationale))
    item.status = "complete"
    append_audit(s, action="review.adjudicated", actor_id=user.id, team_id=user.team_id,
                 subject_type="review_item", subject_id=item.id, payload={"choice": req.choice})
    return {"ok": True}


@app.get("/api/review/stats")
def api_review_stats(batch_id: int | None = None, user: User = Depends(current_user),
                     s: Session = Depends(db)):
    report = reviewlib.agreement_report(s, user.team_id, batch_id)
    pairs = s.execute(
        select(PreferencePair.source, func.count(), func.avg(PreferencePair.margin))
        .where(PreferencePair.team_id == user.team_id).group_by(PreferencePair.source)
    ).all()
    mine = s.execute(
        select(func.count()).select_from(Annotation)
        .where(Annotation.user_id == user.id, Annotation.superseded_by.is_(None))
    ).scalar()
    return {
        **report,
        "my_annotations": mine,
        "pairs": [{"source": src, "count": n, "mean_margin": round(avg or 0, 3)}
                  for src, n, avg in pairs],
        "pairs_total": sum(n for _, n, _ in pairs),
    }


# ------------------------------------------------------------------ playground

@app.post("/api/playground")
def api_playground(req: PlaygroundRequest, user: User = Depends(current_user),
                   s: Session = Depends(db)):
    """Generate once, synchronously. The only slow handler, and deliberately so:
    a reviewer poking at a model wants an answer, not a job id."""
    from .trainers.base import TrainRequest, resolve_backend
    from .trainers.factory import build_policy
    from .trainers.policy import generate, render

    mv = _owned(s, ModelVersion, req.model_version_id, user, "model version") \
        if req.model_version_id else None
    base = mv.base_model if mv else (req.base_model or base_models()[0]["id"])
    backend = resolve_backend("auto", base)

    messages = ([{"role": "system", "content": req.system}] if req.system else []) + \
               [{"role": "user", "content": req.prompt}]
    rows = [{"messages": messages, "completion": ""}]
    stub = TrainRequest(run_id=0, method="playground", base_model=base, params={}, lora={},
                        out_dir=Path("/tmp"), train_rows=rows, val_rows=[])

    policy = build_policy(stub, backend, adapter_dir=mv.artifact_dir if mv else None,
                          inference_only=True)
    prompt = render(policy.tokenizer, messages)
    tuned = generate(policy, [prompt], max_new_tokens=req.max_new_tokens,
                     temperature=req.temperature)[0][0]

    out: dict[str, Any] = {
        "backend": backend, "base_model": base,
        "model": f"{mv.name} v{mv.version}" if mv else base,
        "response": tuned,
    }
    if req.compare_to_base and mv:
        # Same model, adapters switched off — the honest A/B, since any other
        # difference (tokenizer, prompt format, sampling seed) is held constant.
        from .trainers.lora import adapters_disabled
        with adapters_disabled(policy.model):
            out["base_response"] = generate(policy, [prompt],
                                            max_new_tokens=req.max_new_tokens,
                                            temperature=req.temperature)[0][0]
    return out


# ----------------------------------------------------------------------- admin

@app.get("/api/teams")
def api_teams(user: User = Depends(require_admin), s: Session = Depends(db)):
    teams = s.execute(select(Team).order_by(Team.id)).scalars().all()
    members = dict(s.execute(select(User.team_id, func.count()).group_by(User.team_id)).all())
    running = dict(s.execute(
        select(Job.team_id, func.count()).where(Job.status == "running").group_by(Job.team_id)
    ).all())
    return [{"id": t.id, "slug": t.slug, "name": t.name, "concurrency": t.concurrency,
             "members": members.get(t.id, 0), "running": running.get(t.id, 0)} for t in teams]


@app.post("/api/teams")
def api_create_team(req: TeamCreate, user: User = Depends(require_admin),
                    s: Session = Depends(db)):
    if s.execute(select(Team).where(Team.slug == req.slug)).scalar_one_or_none():
        raise HTTPException(409, f"team '{req.slug}' already exists")
    team = Team(slug=req.slug, name=req.name, concurrency=req.concurrency)
    s.add(team)
    s.flush()
    append_audit(s, action="team.created", actor_id=user.id, team_id=team.id,
                 subject_type="team", subject_id=team.id, payload={"slug": req.slug})
    return {"id": team.id, "slug": team.slug, "name": team.name}


@app.post("/api/users")
def api_create_user(req: UserCreate, user: User = Depends(require_admin),
                    s: Session = Depends(db)):
    if s.execute(select(User).where(User.email == req.email.lower())).scalar_one_or_none():
        raise HTTPException(409, "email already registered")
    if s.get(Team, req.team_id) is None:
        raise HTTPException(404, "no such team")
    u = User(email=req.email.lower(), name=req.name, role=req.role, team_id=req.team_id,
             password_hash=authlib.hash_password(req.password))
    s.add(u)
    s.flush()
    append_audit(s, action="user.created", actor_id=user.id, team_id=req.team_id,
                 subject_type="user", subject_id=u.id, payload={"role": req.role})
    return {"id": u.id, "email": u.email, "role": u.role}


@app.get("/api/audit")
def api_audit(limit: int = 100, user: User = Depends(require_lead), s: Session = Depends(db)):
    from .models import AuditEntry
    rows = s.execute(
        select(AuditEntry).where(AuditEntry.team_id == user.team_id)
        .order_by(AuditEntry.seq.desc()).limit(limit)
    ).scalars().all()
    return {
        "entries": [{"seq": e.seq, "ts": e.ts, "action": e.action, "actor_id": e.actor_id,
                     "actor_type": e.actor_type, "subject_type": e.subject_type,
                     "subject_id": e.subject_id, "payload": e.payload} for e in rows],
        "chain": verify_audit_chain(s),
    }


@app.get("/api/health")
def api_health():
    stats = q.queue_stats()
    alive = [w for w in stats["workers"] if w["alive"]]
    return {
        "ok": True,
        "workers_alive": len(alive),
        "queue": stats["by_status"],
        # The single most useful field here: a queue that only grows with no live
        # worker is the failure this endpoint exists to surface.
        "warning": None if alive else "no live workers — jobs will queue indefinitely",
    }


# ------------------------------------------------------------------- static UI

STATIC = static_dir()
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
