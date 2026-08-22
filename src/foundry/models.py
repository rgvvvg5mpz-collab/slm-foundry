"""ORM schema for SLM Foundry.

Design decisions worth knowing before changing anything here:

* **Every row that can belong to a team carries ``team_id`` directly**, even when
  it could be reached by a join. Tenancy filtering that depends on a join is
  tenancy filtering that a future query will forget; a column the API layer
  always filters on is one that is hard to leave out.
* **Annotations are append-only.** A reviewer who changes their mind writes a new
  row and the old one is marked superseded. "What did the reviewer think, and
  when did that change?" is a question a model-governance review will ask, and an
  UPDATE destroys the answer.
* **Jobs own their lease, not their worker.** A worker holds a job by writing a
  random ``lease_token`` and an expiry; every subsequent write is conditional on
  that token. A worker that hangs, gets SIGKILLed, or is partitioned loses the
  job at expiry and cannot later clobber the retry that replaced it.
* **``sampling_prob`` on a review assignment is load-bearing.** The review queue
  is deliberately biased toward items the policy is unsure about. Without the
  propensity that produced each assignment, every rate estimated from reviewed
  data is biased, and a win-rate computed off it means nothing.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# JSONB on Postgres, plain JSON elsewhere (keeps the SQLite dev path working).
JSONType = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


# --------------------------------------------------------------- teams and users

class Team(Base):
    """A tenant. Owns datasets, runs, models, and a slice of the queue."""
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    # Hard cap on simultaneously *running* jobs. The scheduler additionally
    # fair-shares below this cap, so a team that queues 500 jobs at midnight does
    # not starve a team that queues one at 09:00.
    concurrency: Mapped[int] = mapped_column(Integer, default=2)
    gpu_budget_minutes: Mapped[int] = mapped_column(Integer, default=0)  # 0 = untracked
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    # viewer: read-only. member: upload data, launch runs. reviewer: also annotates.
    # lead: also adjudicates disagreements and promotes models. admin: also manages
    # teams, quotas, and users.
    role: Mapped[str] = mapped_column(String(32), default="member", index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    team: Mapped[Team] = relationship()


ROLE_RANK = {"viewer": 0, "member": 1, "reviewer": 2, "lead": 3, "admin": 4}


class ApiKey(Base):
    """A long-lived credential for scripts, applications and CI.

    Session tokens expire in twelve hours, which is right for a browser and
    useless for a nightly job or a service calling the inference endpoint. These
    do not expire on a clock.

    Only the SHA-256 of the secret is stored, so the plaintext exists exactly once
    — in the response that created it. ``prefix`` keeps the first few characters
    in the clear purely so a human can tell two keys apart in a list without
    being able to reconstruct either.
    """
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(200))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    prefix: Mapped[str] = mapped_column(String(16))
    # 'serve' grants only the /v1 inference endpoints; 'full' also grants the
    # management API. Separate because the credential baked into a production
    # service should not be able to delete a dataset.
    scope: Mapped[str] = mapped_column(String(16), default="serve")
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    last_used_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    calls: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    team: Mapped[Team] = relationship()


class AuthSession(Base):
    __tablename__ = "auth_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)

    user: Mapped[User] = relationship()


# ------------------------------------------------------------------- datasets

class Dataset(Base):
    """An uploaded corpus, immutable once it reaches ``ready``.

    Immutability is not fussiness: a run records the dataset id it trained on, and
    a dataset that can be edited afterwards turns every recorded lineage into a
    guess. Corrections create a new version and bump ``version``.
    """
    __tablename__ = "datasets"
    __table_args__ = (
        UniqueConstraint("team_id", "name", "version", name="uq_dataset_name_version"),
        Index("ix_datasets_team_kind", "team_id", "kind", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[int] = mapped_column(Integer, default=1)
    # sft | preference | prompts | benchmark
    kind: Mapped[str] = mapped_column(String(32), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    # validating | ready | invalid
    status: Mapped[str] = mapped_column(String(32), default="validating", index=True)
    path: Mapped[str] = mapped_column(String(1024), default="")
    sha256: Mapped[str] = mapped_column(String(64), default="", index=True)
    num_rows: Mapped[int] = mapped_column(Integer, default=0)
    num_bad_rows: Mapped[int] = mapped_column(Integer, default=0)
    bytes: Mapped[int] = mapped_column(Integer, default=0)
    val_fraction: Mapped[float] = mapped_column(Float, default=0.1)
    stats: Mapped[dict] = mapped_column(JSONType, default=dict)
    errors: Mapped[list] = mapped_column(JSONType, default=list)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    team: Mapped[Team] = relationship()


# ----------------------------------------------------------------------- runs

class Run(Base):
    """One training job's configuration and outcome.

    ``parent_model_version_id`` is what makes the flywheel legible: an SFT run has
    none, a DPO run points at the SFT model it improved, and the registry can walk
    the chain backwards to answer "what produced this?" without reading logs.
    """
    __tablename__ = "runs"
    __table_args__ = (Index("ix_runs_team_status", "team_id", "status"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    # sft | reward | dpo | ppo | grpo | gspo | rlaif
    method: Mapped[str] = mapped_column(String(32), index=True)
    base_model: Mapped[str] = mapped_column(String(200))
    parent_model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True, index=True)
    reward_model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True)
    train_dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    eval_dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    review_batch_id: Mapped[int | None] = mapped_column(ForeignKey("review_batches.id"), nullable=True)
    # Fully resolved: defaults filled, values clamped. Never the raw submission.
    params: Mapped[dict] = mapped_column(JSONType, default=dict)
    lora: Mapped[dict] = mapped_column(JSONType, default=dict)
    backend: Mapped[str] = mapped_column(String(16), default="auto")
    # queued | running | succeeded | failed | cancelled
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    metrics: Mapped[dict] = mapped_column(JSONType, default=dict)
    artifact_dir: Mapped[str] = mapped_column(String(1024), default="")
    error: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    team: Mapped[Team] = relationship()


# ----------------------------------------------------------------------- queue

class Job(Base):
    """A unit of background work with an expiring lease.

    Status transitions, and the only legal ones:

        queued  ──claim──▶  running  ──finish──▶  succeeded
           ▲                   │      ──fail────▶  failed  (attempts exhausted)
           └────requeue────────┘                            │
              (lease expiry, or a retryable failure)  ◀──────┘

    ``cancel_requested`` is a flag rather than a status because a running job can
    only be stopped cooperatively — the trainer checks it between steps. Setting
    the status directly would strand the worker writing into a job it no longer owns.
    """
    __tablename__ = "jobs"
    __table_args__ = (
        Index("ix_jobs_claim", "status", "kind", "priority", "id"),
        Index("ix_jobs_team_status", "team_id", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    # train | eval | generate | judge | assemble
    kind: Mapped[str] = mapped_column(String(32), index=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id"), nullable=True, index=True)
    payload: Mapped[dict] = mapped_column(JSONType, default=dict)
    # Higher runs first. Interactive work (a reviewer waiting on candidate
    # generation) outranks a batch retrain nobody is watching.
    priority: Mapped[int] = mapped_column(Integer, default=0, index=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    lease_token: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_expires_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    progress: Mapped[dict] = mapped_column(JSONType, default=dict)
    error: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    team: Mapped[Team] = relationship()


class JobEvent(Base):
    """The live log line stream for a job. Cheap, append-only, tail-able."""
    __tablename__ = "job_events"
    __table_args__ = (Index("ix_job_events_job_seq", "job_id", "id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"), index=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    level: Mapped[str] = mapped_column(String(16), default="info")   # debug|info|warn|error|metric
    message: Mapped[str] = mapped_column(Text, default="")
    data: Mapped[dict] = mapped_column(JSONType, default=dict)


class Worker(Base):
    """Registered worker processes. Purely observability — the queue does not
    consult this table to hand out work; leases do that."""
    __tablename__ = "workers"

    id: Mapped[int] = mapped_column(primary_key=True)
    worker_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    hostname: Mapped[str] = mapped_column(String(255), default="")
    kinds: Mapped[list] = mapped_column(JSONType, default=list)
    capabilities: Mapped[dict] = mapped_column(JSONType, default=dict)
    current_job_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


# ------------------------------------------------------------- model registry

class ModelVersion(Base):
    """A trained artifact, versioned per (team, name).

    Lifecycle: ``draft`` → ``evaluated`` → ``staging`` → ``production`` → ``archived``.
    Only one production version per (team, name) at a time; promoting demotes the
    incumbent to ``archived`` in the same transaction.
    """
    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("team_id", "name", "version", name="uq_model_name_version"),
        Index("ix_model_versions_team_status", "team_id", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    name: Mapped[str] = mapped_column(String(200), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id"), nullable=True)
    base_model: Mapped[str] = mapped_column(String(200))
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"), nullable=True)
    method: Mapped[str] = mapped_column(String(32), default="sft")
    # adapter | reward | full
    artifact_kind: Mapped[str] = mapped_column(String(32), default="adapter")
    artifact_dir: Mapped[str] = mapped_column(String(1024), default="")
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)
    metrics: Mapped[dict] = mapped_column(JSONType, default=dict)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    promoted_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    promoted_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    team: Mapped[Team] = relationship()


class ModelExport(Base):
    """A downloadable package built from a model version.

    ``adapter`` is the LoRA weights alone — a few megabytes, useless without the
    base model. ``merged`` folds them into the base weights and writes a complete
    model directory that any transformers-compatible runtime can load with no
    knowledge of this system. The second is what people actually want when they
    are taking a model somewhere else, and it is slow enough to need a job.
    """
    __tablename__ = "model_exports"

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), index=True)
    job_id: Mapped[int | None] = mapped_column(ForeignKey("jobs.id"), nullable=True)
    # adapter | merged
    fmt: Mapped[str] = mapped_column(String(16), default="adapter")
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    path: Mapped[str] = mapped_column(String(1024), default="")
    bytes: Mapped[int] = mapped_column(Integer, default=0)
    sha256: Mapped[str] = mapped_column(String(64), default="")
    error: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EvalRun(Base):
    """One model version scored against one benchmark dataset."""
    __tablename__ = "eval_runs"
    __table_args__ = (
        UniqueConstraint("model_version_id", "dataset_id", name="uq_eval_model_dataset"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), index=True)
    job_id: Mapped[int | None] = mapped_column(ForeignKey("jobs.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    metrics: Mapped[dict] = mapped_column(JSONType, default=dict)
    per_example_path: Mapped[str] = mapped_column(String(1024), default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ------------------------------------------------------------------- review

class ReviewBatch(Base):
    """A campaign: sample prompts, generate candidates, collect judgements.

    A batch pins the policy version it sampled from. Preferences collected against
    one policy and used to train a different one are off-policy data — sometimes
    fine, but the pipeline should never *silently* do it, so the id is recorded.
    """
    __tablename__ = "review_batches"

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    # pairwise | rank | rubric
    protocol: Mapped[str] = mapped_column(String(32), default="pairwise")
    policy_model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True)
    prompt_dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    candidates_per_prompt: Mapped[int] = mapped_column(Integer, default=2)
    annotations_per_item: Mapped[int] = mapped_column(Integer, default=1)
    # Fraction of items also sent to the RLAIF judge. At 1.0 the batch is pure
    # RLAIF; at 0.1 the judge is being audited against humans on a sample.
    ai_assist_fraction: Mapped[float] = mapped_column(Float, default=0.0)
    judge_model: Mapped[str] = mapped_column(String(64), default="")
    # draft | generating | open | closed
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReviewItem(Base):
    """One prompt plus its candidate responses, awaiting judgement."""
    __tablename__ = "review_items"
    __table_args__ = (Index("ix_review_items_queue", "batch_id", "status", "priority"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    batch_id: Mapped[int] = mapped_column(ForeignKey("review_batches.id"), index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    prompt: Mapped[str] = mapped_column(Text)
    # [{"id": "a", "text": ..., "logprob": ..., "source": "policy"|"reference"|"dataset"}, ...]
    candidates: Mapped[list] = mapped_column(JSONType, default=list)
    # Model-side uncertainty used to bias sampling toward informative items.
    uncertainty: Mapped[float] = mapped_column(Float, default=0.0)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    # pending | in_review | complete | skipped | disputed
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    meta: Mapped[dict] = mapped_column(JSONType, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReviewAssignment(Base):
    __tablename__ = "review_assignments"
    __table_args__ = (
        UniqueConstraint("item_id", "user_id", name="uq_assignment_item_user"),
        Index("ix_assignments_user_status", "user_id", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("review_items.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    # See the module docstring: this is statistics, not telemetry.
    sampling_prob: Mapped[float] = mapped_column(Float, default=1.0)
    # assigned | done | skipped | expired
    status: Mapped[str] = mapped_column(String(32), default="assigned", index=True)
    assigned_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Annotation(Base):
    """One judgement. Append-only — corrections supersede, never overwrite."""
    __tablename__ = "annotations"
    __table_args__ = (Index("ix_annotations_item", "item_id", "superseded_by"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("review_items.id"), index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    # 'human' or 'ai'. Explicit rather than inferred from a null user_id, because
    # the governance views query it constantly and should never need a join.
    annotator_type: Mapped[str] = mapped_column(String(16), default="human", index=True)
    judge_model: Mapped[str] = mapped_column(String(64), default="")
    protocol: Mapped[str] = mapped_column(String(32), default="pairwise")
    # pairwise: the winning candidate id, or "tie" / "both_bad"
    choice: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # rank: ordered candidate ids, best first
    ranking: Mapped[list] = mapped_column(JSONType, default=list)
    # rubric: {"helpfulness": {"a": 4, "b": 2}, ...}
    rubric_scores: Mapped[dict] = mapped_column(JSONType, default=dict)
    rationale: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    superseded_by: Mapped[int | None] = mapped_column(ForeignKey("annotations.id"), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Adjudication(Base):
    """A lead's tie-break when annotators disagree. Final, and it wins."""
    __tablename__ = "adjudications"

    id: Mapped[int] = mapped_column(primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("review_items.id"), unique=True, index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    choice: Mapped[str] = mapped_column(String(32))
    rationale: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PreferencePair(Base):
    """The consumable output of review: what the optimiser actually trains on.

    Materialised rather than derived on the fly so that a training run records
    exactly which pairs it saw. Re-deriving from annotations later would give a
    different set as soon as one more reviewer submits.
    """
    __tablename__ = "preference_pairs"
    __table_args__ = (Index("ix_pairs_team_batch", "team_id", "batch_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    batch_id: Mapped[int | None] = mapped_column(ForeignKey("review_batches.id"), nullable=True)
    item_id: Mapped[int | None] = mapped_column(ForeignKey("review_items.id"), nullable=True)
    prompt: Mapped[str] = mapped_column(Text)
    chosen: Mapped[str] = mapped_column(Text)
    rejected: Mapped[str] = mapped_column(Text)
    # human | ai | mixed
    source: Mapped[str] = mapped_column(String(16), default="human", index=True)
    # Strength of preference in [0,1]: unanimous humans or a wide judge gap → 1.0.
    # DPO can weight by it; a 3-2 split should not push as hard as a 5-0.
    margin: Mapped[float] = mapped_column(Float, default=1.0)
    agreement: Mapped[float | None] = mapped_column(Float, nullable=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# -------------------------------------------------------------------- audit

class AuditEntry(Base):
    """Hash-chained record of every consequential action.

    Each entry commits to its predecessor's hash, so a row cannot be altered or
    removed without breaking every hash after it. Cheap to write, and it turns
    "who promoted this model to production?" from a log-grep into a query.
    """
    __tablename__ = "audit_entries"

    id: Mapped[int] = mapped_column(primary_key=True)
    seq: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"), nullable=True, index=True)
    actor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    actor_type: Mapped[str] = mapped_column(String(16), default="user")
    action: Mapped[str] = mapped_column(String(64), index=True)
    subject_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    subject_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload: Mapped[dict] = mapped_column(JSONType, default=dict)
    prev_hash: Mapped[str] = mapped_column(String(64), default="")
    entry_hash: Mapped[str] = mapped_column(String(64), default="", index=True)
