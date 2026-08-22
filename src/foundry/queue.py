"""The job queue: fair-share scheduling over a database table.

Why a table and not Redis/Celery: the jobs here are minutes-to-hours long, arrive
a few per minute at most, and their state is something six teams need to *query*
("show me my team's runs, oldest first, with progress"). A broker optimised for
100k messages/second buys nothing at this volume and costs an extra service to
operate, plus a second source of truth that drifts from the one the UI reads.

Two properties do have to be right, and both are:

**Exactly-once claim.** A worker takes a job with a conditional UPDATE — set my
lease token *where status is still queued*. Two workers racing on the same row
produce one rowcount=1 and one rowcount=0; the loser moves to the next candidate.
No locking protocol, no broker, correct on SQLite and Postgres alike.

**No permanently stuck jobs.** A claim writes an expiry, and every subsequent
write by that worker is conditional on still holding the token. A worker that is
SIGKILLed, OOMs, or gets partitioned simply stops renewing; the reaper requeues
the job at expiry, and the zombie's late writes hit `rowcount=0` and are dropped.

Fair share is deliberately *not* strict priority. Strict priority lets one team's
overnight sweep of 400 jobs monopolise every worker. Candidates are ordered by
how many jobs the team is already running first, and only then by priority — so a
team with nothing running always gets the next free worker.
"""
from __future__ import annotations

import datetime as dt
import secrets
import socket
from typing import Any, Iterable

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from .config import queue_config
from .db import get_session
from .models import Job, JobEvent, Team, Worker, utcnow

TERMINAL = ("succeeded", "failed", "cancelled")


# ------------------------------------------------------------------- producing

def enqueue(
    s: Session,
    *,
    team_id: int,
    kind: str,
    payload: dict[str, Any] | None = None,
    run_id: int | None = None,
    priority: int = 0,
    created_by: int | None = None,
    max_attempts: int | None = None,
) -> Job:
    job = Job(
        team_id=team_id, kind=kind, run_id=run_id, payload=payload or {},
        priority=priority, created_by=created_by,
        max_attempts=max_attempts or queue_config()["max_attempts"],
    )
    s.add(job)
    s.flush()
    return job


# ------------------------------------------------------------------- consuming

def claim(worker_id: str, kinds: Iterable[str], lease_seconds: int | None = None) -> dict | None:
    """Take one job for this worker, or return None if there is nothing to do.

    Returns a plain dict rather than an ORM object: the caller runs for minutes
    afterwards and must not be holding a Session open across that. Everything the
    worker needs is copied out here.
    """
    lease_seconds = lease_seconds or queue_config()["lease_seconds"]
    kinds = list(kinds)

    with get_session() as s:
        # Concurrency already in flight, per team.
        running: dict[int, int] = {
            tid: n for tid, n in s.execute(
                select(Job.team_id, func.count()).where(Job.status == "running").group_by(Job.team_id)
            ).all()
        }
        caps: dict[int, int] = {
            t.id: t.concurrency for t in s.execute(select(Team)).scalars()
        }
        default_cap = queue_config()["default_team_concurrency"]

        candidates = s.execute(
            select(Job)
            .where(Job.status == "queued", Job.kind.in_(kinds), Job.cancel_requested.is_(False))
            .order_by(Job.priority.desc(), Job.id.asc())
            .limit(200)
        ).scalars().all()

        eligible = [
            j for j in candidates
            if running.get(j.team_id, 0) < caps.get(j.team_id, default_cap)
        ]
        # Fewest running first — that is the fair-share bit — then priority, then FIFO.
        eligible.sort(key=lambda j: (running.get(j.team_id, 0), -j.priority, j.id))

        for job in eligible:
            token = secrets.token_hex(16)
            res = s.execute(
                update(Job)
                .where(Job.id == job.id, Job.status == "queued")
                .values(
                    status="running", lease_token=token, worker_id=worker_id,
                    lease_expires_at=utcnow() + dt.timedelta(seconds=lease_seconds),
                    attempts=Job.attempts + 1, started_at=utcnow(),
                )
            )
            if res.rowcount == 1:                      # we won the race for this row
                s.flush()
                fresh = s.get(Job, job.id)
                s.add(JobEvent(job_id=job.id, level="info",
                               message=f"claimed by {worker_id}",
                               data={"attempt": fresh.attempts}))
                return {
                    "id": fresh.id, "team_id": fresh.team_id, "kind": fresh.kind,
                    "run_id": fresh.run_id, "payload": dict(fresh.payload or {}),
                    "attempts": fresh.attempts, "max_attempts": fresh.max_attempts,
                    "lease_token": token,
                }
    return None


def heartbeat(job_id: int, token: str, progress: dict[str, Any] | None = None) -> bool:
    """Extend the lease. Returns False if we no longer hold the job.

    A False here means the worker was declared dead and something else owns this
    job now — the correct response is to abandon the work immediately rather than
    keep computing into an artifact directory someone else is writing.
    """
    lease = queue_config()["lease_seconds"]
    values: dict[str, Any] = {"lease_expires_at": utcnow() + dt.timedelta(seconds=lease)}
    if progress is not None:
        values["progress"] = progress
    with get_session() as s:
        res = s.execute(
            update(Job)
            .where(Job.id == job_id, Job.lease_token == token, Job.status == "running")
            .values(**values)
        )
        return res.rowcount == 1


def is_cancelled(job_id: int) -> bool:
    with get_session() as s:
        job = s.get(Job, job_id)
        return bool(job and (job.cancel_requested or job.status == "cancelled"))


def finish(job_id: int, token: str, *, status: str, error: str = "",
           progress: dict[str, Any] | None = None) -> bool:
    values: dict[str, Any] = {
        "status": status, "finished_at": utcnow(), "error": error[:8000],
        "lease_token": None, "lease_expires_at": None,
    }
    if progress is not None:
        values["progress"] = progress
    with get_session() as s:
        res = s.execute(
            update(Job).where(Job.id == job_id, Job.lease_token == token).values(**values)
        )
        if res.rowcount == 1:
            s.add(JobEvent(job_id=job_id, level="info" if status == "succeeded" else "error",
                           message=f"job {status}", data={"error": error[:500]} if error else {}))
        return res.rowcount == 1


def fail_or_retry(job_id: int, token: str, error: str) -> str:
    """Requeue if attempts remain, otherwise fail. Returns the resulting status."""
    with get_session() as s:
        job = s.get(Job, job_id)
        if job is None or job.lease_token != token:
            return "lost"
        if job.cancel_requested:
            status = "cancelled"
        elif job.attempts < job.max_attempts:
            status = "queued"
        else:
            status = "failed"

        job.status = status
        job.error = error[:8000]
        job.lease_token = None
        job.lease_expires_at = None
        if status != "queued":
            job.finished_at = utcnow()
        s.add(JobEvent(
            job_id=job_id, level="warn" if status == "queued" else "error",
            message=f"attempt {job.attempts}/{job.max_attempts} failed → {status}",
            data={"error": error[:1000]},
        ))
        return status


def request_cancel(s: Session, job_id: int) -> bool:
    """Ask a job to stop. Queued jobs die immediately; running ones stop at the
    next step boundary — a trainer killed mid-optimiser-step leaves a torn
    checkpoint, so cancellation is cooperative by design."""
    job = s.get(Job, job_id)
    if job is None or job.status in TERMINAL:
        return False
    job.cancel_requested = True
    if job.status == "queued":
        job.status = "cancelled"
        job.finished_at = utcnow()
    s.add(JobEvent(job_id=job_id, level="warn", message="cancellation requested"))
    return True


# --------------------------------------------------------------------- reaping

def reap_expired() -> int:
    """Requeue jobs whose lease lapsed. Idempotent; safe to run from every worker."""
    now = utcnow()
    requeued = 0
    with get_session() as s:
        stale = s.execute(
            select(Job).where(Job.status == "running", Job.lease_expires_at < now)
        ).scalars().all()
        for job in stale:
            exhausted = job.attempts >= job.max_attempts
            job.status = "failed" if exhausted else "queued"
            job.lease_token = None                     # invalidates the zombie's writes
            job.lease_expires_at = None
            if exhausted:
                job.error = "lease expired; attempts exhausted"
                job.finished_at = now
            s.add(JobEvent(
                job_id=job.id, level="warn",
                message=f"lease expired on {job.worker_id} → {job.status}",
            ))
            requeued += 1
    return requeued


# --------------------------------------------------------------------- logging

def log(job_id: int, message: str, level: str = "info", **data: Any) -> None:
    with get_session() as s:
        s.add(JobEvent(job_id=job_id, level=level, message=message[:4000], data=data))


# ------------------------------------------------------------- worker registry

def register_worker(worker_id: str, kinds: list[str], capabilities: dict[str, Any]) -> None:
    with get_session() as s:
        w = s.execute(select(Worker).where(Worker.worker_id == worker_id)).scalar_one_or_none()
        if w is None:
            w = Worker(worker_id=worker_id, hostname=socket.gethostname())
            s.add(w)
        w.kinds = kinds
        w.capabilities = capabilities
        w.last_seen_at = utcnow()


def touch_worker(worker_id: str, current_job_id: int | None) -> None:
    with get_session() as s:
        w = s.execute(select(Worker).where(Worker.worker_id == worker_id)).scalar_one_or_none()
        if w:
            w.last_seen_at = utcnow()
            w.current_job_id = current_job_id


def queue_stats() -> dict[str, Any]:
    with get_session() as s:
        by_status = dict(s.execute(select(Job.status, func.count()).group_by(Job.status)).all())
        by_team = [
            {"team_id": tid, "status": st, "count": n}
            for tid, st, n in s.execute(
                select(Job.team_id, Job.status, func.count())
                .where(Job.status.in_(("queued", "running")))
                .group_by(Job.team_id, Job.status)
            ).all()
        ]
        cutoff = utcnow() - dt.timedelta(seconds=queue_config()["heartbeat_seconds"] * 3)
        workers = s.execute(select(Worker).order_by(Worker.worker_id)).scalars().all()
        return {
            "by_status": by_status,
            "by_team": by_team,
            "workers": [
                {
                    "worker_id": w.worker_id, "hostname": w.hostname, "kinds": w.kinds,
                    "current_job_id": w.current_job_id,
                    "capabilities": w.capabilities,
                    "alive": (w.last_seen_at.replace(tzinfo=dt.timezone.utc)
                              if w.last_seen_at.tzinfo is None else w.last_seen_at) > cutoff,
                    "last_seen_at": w.last_seen_at,
                }
                for w in workers
            ],
        }
