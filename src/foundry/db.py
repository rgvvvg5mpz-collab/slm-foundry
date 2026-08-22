"""Engine, session management, and the audit chain.

SQLite is the zero-setup default so a team can try the product with one command;
Postgres is what it should run on with more than one worker. Two places care
about the difference and both handle it explicitly: the audit chain takes an
advisory lock on Postgres (SQLite serialises writes anyway), and the queue's
claim is written as a conditional UPDATE that is correct on both.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from typing import Any, Iterator

from sqlalchemy import create_engine, event, func, select, text
from sqlalchemy.orm import Session, sessionmaker

from .config import database_url
from .models import AuditEntry, Base

_engine = None
_SessionLocal = None


def engine():
    global _engine
    if _engine is None:
        url = database_url()
        kwargs: dict[str, Any] = {"future": True, "pool_pre_ping": True}
        if url.startswith("sqlite"):
            # check_same_thread off: uvicorn serves requests on a threadpool.
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        else:
            kwargs |= {"pool_size": 10, "max_overflow": 10}
        _engine = create_engine(url, **kwargs)

        if url.startswith("sqlite"):
            @event.listens_for(_engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _rec):
                cur = dbapi_conn.cursor()
                # WAL is what lets a worker write progress while the API reads it.
                # Without it the two block each other and the UI stalls mid-run.
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA busy_timeout=30000")
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()
    return _engine


def session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=engine(), expire_on_commit=False, future=True)
    return _SessionLocal


@contextmanager
def get_session() -> Iterator[Session]:
    s = session_factory()()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def init_db(drop: bool = False) -> None:
    if drop:
        Base.metadata.drop_all(engine())
    Base.metadata.create_all(engine())


def reset_engine() -> None:
    """Drop cached engine/session factory. Tests point at a fresh database."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


# -------------------------------------------------------------------- audit chain

def _digest(prev_hash: str, body: dict[str, Any]) -> str:
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(f"{prev_hash}\n{payload}".encode()).hexdigest()


def append_audit(
    s: Session,
    *,
    action: str,
    actor_id: int | None,
    actor_type: str = "user",
    team_id: int | None = None,
    payload: dict[str, Any] | None = None,
    subject_type: str | None = None,
    subject_id: str | int | None = None,
) -> AuditEntry:
    """Append one hash-chained entry. Must run inside a transaction.

    Concurrent appenders must not both read the same head and write sibling
    entries claiming the same predecessor — a fork is indistinguishable from
    tampering once someone verifies the chain. On Postgres an advisory lock
    serialises the read-then-insert; on SQLite the write lock already does.
    """
    if not database_url().startswith("sqlite"):
        s.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": 0x5F00D})

    head = s.execute(
        select(AuditEntry).order_by(AuditEntry.seq.desc()).limit(1)
    ).scalar_one_or_none()
    seq = (head.seq + 1) if head else 1
    prev = head.entry_hash if head else ""

    body = {
        "seq": seq, "action": action, "actor_id": actor_id, "actor_type": actor_type,
        "team_id": team_id, "subject_type": subject_type,
        "subject_id": None if subject_id is None else str(subject_id),
        "payload": payload or {},
    }
    entry = AuditEntry(
        seq=seq, action=action, actor_id=actor_id, actor_type=actor_type,
        team_id=team_id, subject_type=subject_type,
        subject_id=None if subject_id is None else str(subject_id),
        payload=payload or {}, prev_hash=prev, entry_hash=_digest(prev, body),
    )
    s.add(entry)
    s.flush()
    return entry


def verify_audit_chain(s: Session, limit: int | None = None) -> dict[str, Any]:
    q = select(AuditEntry).order_by(AuditEntry.seq.asc())
    if limit:
        q = q.limit(limit)
    prev = ""
    checked = 0
    for e in s.execute(q).scalars():
        body = {
            "seq": e.seq, "action": e.action, "actor_id": e.actor_id,
            "actor_type": e.actor_type, "team_id": e.team_id,
            "subject_type": e.subject_type, "subject_id": e.subject_id,
            "payload": e.payload or {},
        }
        if e.prev_hash != prev or _digest(prev, body) != e.entry_hash:
            return {"ok": False, "broken_at_seq": e.seq, "checked": checked}
        prev = e.entry_hash
        checked += 1
    return {"ok": True, "checked": checked, "head": prev}
