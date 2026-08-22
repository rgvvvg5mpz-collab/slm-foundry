"""The worker: claim a job, do the work, record what happened.

This is the only module that knows both the database and the trainers, and it is
deliberately the only one. Trainers take plain data and emit plain results;
everything about runs, datasets, model versions and review batches is resolved
here and handed over. The result is that a trainer can be exercised from a test
with four lines and no database, and that adding a job kind is a function plus a
dispatch entry.

Run it with::

    python -m foundry.worker --kinds train,eval,generate,judge,assemble

Several workers may run against the same database; the queue's lease protocol
(see :mod:`foundry.queue`) is what keeps them from colliding. Run a dedicated
worker with ``--kinds generate,judge`` next to a GPU box doing ``--kinds train``
and interactive review work stops queueing behind overnight retrains.
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import select

from . import queue as q
from . import registry, review as reviewlib
from .config import judge_config, queue_config
from .datasets import load_rows, split_rows
from .db import append_audit, get_session, init_db
from .evaluate import evaluate_rows
from .judge import get_judge
from .models import (
    Dataset, EvalRun, ModelVersion, PreferencePair, ReviewBatch, ReviewItem, Run, utcnow,
)
from .paths import eval_dir, run_dir
from .trainers import Cancelled, TrainContext, TrainRequest, get_trainer
from .trainers.base import resolve_backend

ALL_KINDS = ("train", "eval", "generate", "judge", "assemble")
_shutdown = False


# --------------------------------------------------------------------- helpers

def _ctx_for(job: dict, out_dir: Path) -> TrainContext:
    job_id, token = job["id"], job["lease_token"]
    return TrainContext(
        out_dir=out_dir,
        on_log=lambda level, msg, data: q.log(job_id, msg, level, **data),
        on_progress=lambda p: q.heartbeat(job_id, token, p),
        is_cancelled=lambda: q.is_cancelled(job_id),
    )


def _dataset_rows(s, dataset_id: int | None) -> list[dict]:
    if not dataset_id:
        return []
    ds = s.get(Dataset, dataset_id)
    if ds is None or ds.status != "ready":
        raise ValueError(f"dataset {dataset_id} is not ready")
    return load_rows(ds.path)


def _pairs_from_review(s, team_id: int, batch_id: int | None) -> list[dict]:
    """Preference pairs collected in the review console, as trainer rows."""
    query = select(PreferencePair).where(PreferencePair.team_id == team_id)
    if batch_id:
        query = query.where(PreferencePair.batch_id == batch_id)
    return [
        {
            "messages": [{"role": "user", "content": p.prompt}],
            "chosen": p.chosen, "rejected": p.rejected,
            # Reviewer consensus and queue propensity both fold into one number
            # here, because that is the only knob the loss actually has.
            "margin": max(0.05, min(1.0, p.margin * p.weight)),
            "source": p.source,
        }
        for p in s.execute(query).scalars()
    ]


def _load_reward_fn(s, run: Run, rows: list[dict], backend: str, ctx: TrainContext):
    """Pick a reward source for an RL run and say which one was chosen."""
    from .rewards import from_judge, from_reward_model, from_verifier, references_from_rows

    if run.reward_model_version_id:
        mv = s.get(ModelVersion, run.reward_model_version_id)
        if mv is None or mv.artifact_kind != "reward":
            raise ValueError("reward_model_version_id does not point at a reward model")
        from .trainers.factory import build_policy
        from .trainers.policy import RewardModel
        stub = TrainRequest(run_id=run.id, method="reward", base_model=mv.base_model,
                            params={}, lora=run.lora, out_dir=ctx.out_dir,
                            train_rows=rows, val_rows=[])
        policy = build_policy(stub, backend, adapter_dir=mv.artifact_dir, inference_only=True)
        rm = RewardModel(policy)
        try:
            rm.load_head(mv.artifact_dir)
        except Exception as e:
            ctx.log(f"reward head not loaded ({e}); scores are from an untrained head",
                    level="warn")
        ctx.log(f"reward source: reward model v{mv.version} (#{mv.id})")
        return from_reward_model(rm)

    from .trainers.factory import build_policy, prompt_texts
    stub = TrainRequest(run_id=run.id, method=run.method, base_model=run.base_model,
                        params={}, lora=run.lora, out_dir=ctx.out_dir,
                        train_rows=rows, val_rows=[])
    probe = build_policy(stub, backend, inference_only=True)
    refs = references_from_rows(prompt_texts(probe, rows), rows)
    del probe

    if refs and len(refs) >= max(1, len(rows) // 2):
        # A checkable answer beats a learned scorer: it cannot be gamed and it
        # costs nothing per rollout.
        ctx.log(f"reward source: verifier over {len(refs)} reference answers")
        return from_verifier(refs)

    judge = get_judge()
    ctx.log(f"reward source: judge model {getattr(judge, 'model', '?')}"
            + (" (mechanical fallback — no API key)"
               if getattr(judge, "model", "") == "heuristic-v1" else ""),
            level="warn" if getattr(judge, "model", "") == "heuristic-v1" else "info")
    return from_judge(judge, refs)


# ------------------------------------------------------------------- job: train

def handle_train(job: dict) -> dict[str, Any]:
    with get_session() as s:
        run = s.get(Run, job["run_id"])
        if run is None:
            raise ValueError(f"no run {job['run_id']}")
        run.status = "running"
        run.started_at = utcnow()
        team_id, method = run.team_id, run.method
        run_snapshot = {
            "id": run.id, "name": run.name, "method": method,
            "base_model": run.base_model, "params": dict(run.params),
            "lora": dict(run.lora), "backend": run.backend,
            "train_dataset_id": run.train_dataset_id,
            "eval_dataset_id": run.eval_dataset_id,
            "review_batch_id": run.review_batch_id,
            "parent_model_version_id": run.parent_model_version_id,
            "created_by": run.created_by,
        }

    out = run_dir(run_snapshot["id"])
    ctx = _ctx_for(job, out)
    backend = resolve_backend(run_snapshot["backend"], run_snapshot["base_model"])
    if backend == "tiny":
        ctx.log(f"{run_snapshot['base_model']} is not available locally — running on the "
                f"tiny backend: identical training maths, a randomly-initialised "
                f"two-layer model. Metrics are real; quality numbers are not.", level="warn")

    # ---- assemble the training rows -----------------------------------------
    with get_session() as s:
        if method in ("dpo", "reward"):
            rows = _dataset_rows(s, run_snapshot["train_dataset_id"])
            if not rows:
                rows = _pairs_from_review(s, team_id, run_snapshot["review_batch_id"])
                ctx.log(f"using {len(rows)} preference pairs from the review console")
        else:
            rows = _dataset_rows(s, run_snapshot["train_dataset_id"])

        parent_adapter = None
        if run_snapshot["parent_model_version_id"]:
            mv = s.get(ModelVersion, run_snapshot["parent_model_version_id"])
            parent_adapter = mv.artifact_dir if mv else None
            if parent_adapter:
                ctx.log(f"starting from {mv.name} v{mv.version}")

    if not rows:
        raise ValueError("no training data — attach a dataset or collect a review batch")

    val_fraction = 0.1 if method in ("sft", "dpo", "reward") else 0.0
    train_rows, val_rows = split_rows(rows, val_fraction, seed=0)

    req = TrainRequest(
        run_id=run_snapshot["id"], method=method, base_model=run_snapshot["base_model"],
        params=run_snapshot["params"], lora=run_snapshot["lora"], out_dir=out,
        backend=backend, train_rows=train_rows, val_rows=val_rows,
        policy_adapter_dir=parent_adapter,
        extra={"max_seq_len": int(run_snapshot["params"].get("max_seq_len", 1024))},
    )

    if method in ("ppo", "grpo", "gspo"):
        with get_session() as s:
            run = s.get(Run, run_snapshot["id"])
            req.reward_fn = _load_reward_fn(s, run, train_rows, backend, ctx)

    ctx.write_json("config.json", {**run_snapshot, "resolved_backend": backend,
                                   "train_rows": len(train_rows), "val_rows": len(val_rows)})

    result = get_trainer(method)(ctx, req, backend)

    # ---- record the outcome --------------------------------------------------
    with get_session() as s:
        run = s.get(Run, run_snapshot["id"])
        run.status = "succeeded"
        run.finished_at = utcnow()
        run.metrics = result.metrics
        run.artifact_dir = result.artifact_dir
        run.backend = result.backend

        mv = registry.register(s, run=run, artifact_dir=result.artifact_dir,
                               artifact_kind=result.artifact_kind, metrics=result.metrics)
        followups = []
        if run_snapshot["eval_dataset_id"] and result.artifact_kind == "adapter":
            ev = EvalRun(team_id=team_id, model_version_id=mv.id,
                         dataset_id=run_snapshot["eval_dataset_id"], status="queued")
            s.add(ev)
            s.flush()
            # Benchmarking outranks the next training job: it is what unblocks a
            # human deciding whether to promote.
            ejob = q.enqueue(s, team_id=team_id, kind="eval", payload={"eval_run_id": ev.id},
                             run_id=run.id, priority=5, created_by=run.created_by)
            followups.append({"eval_job": ejob.id})

        # Only the DPO branch of RLAIF produces preference pairs; the GRPO/GSPO
        # branches return rollouts, which have no chosen/rejected to show a human.
        spot = [p for p in (result.samples or [])
                if isinstance(p, dict) and "chosen" in p and "rejected" in p]
        if method == "rlaif" and spot:
            bid = _queue_spot_check(s, run, spot)
            followups.append({"spot_check_batch": bid, "pairs": len(spot)})

        append_audit(s, action="run.succeeded", actor_id=run.created_by, actor_type="worker",
                     team_id=team_id, subject_type="run", subject_id=run.id,
                     payload={"method": method, "backend": result.backend,
                              "model_version_id": mv.id})

    return {"model_version_id": mv.id, "metrics": result.metrics, "followups": followups}


def _queue_spot_check(s, run: Run, pairs: list[dict]) -> int:
    """Put a sample of judge-labelled pairs in front of humans.

    Without this the judge is unfalsifiable: nothing in the system would ever
    disagree with it. The batch it creates flows through the normal review queue,
    and the agreement report compares the two.
    """
    batch = ReviewBatch(
        team_id=run.team_id, name=f"RLAIF spot-check — {run.name}",
        protocol="pairwise", candidates_per_prompt=2, annotations_per_item=1,
        ai_assist_fraction=1.0, judge_model=run.metrics.get("judge_model", ""),
        status="open", created_by=run.created_by,
    )
    s.add(batch)
    s.flush()
    for p in pairs:
        prompt = "\n\n".join(m["content"] for m in p.get("messages", []))
        s.add(ReviewItem(
            batch_id=batch.id, team_id=run.team_id, prompt=prompt,
            # Presented A/B without which side the judge picked — a reviewer shown
            # the judge's answer is checking a box, not making a judgement.
            candidates=[{"id": "a", "text": p["chosen"], "source": "judge_chosen"},
                        {"id": "b", "text": p["rejected"], "source": "judge_rejected"}],
            uncertainty=1.0 - float(p.get("margin", 1.0)), priority=3,
            meta={"run_id": run.id, "judge_margin": p.get("margin"),
                  "judge_rationale": p.get("rationale", "")[:500]},
        ))
    return batch.id


# -------------------------------------------------------------------- job: eval

def handle_eval(job: dict) -> dict[str, Any]:
    with get_session() as s:
        ev = s.get(EvalRun, job["payload"]["eval_run_id"])
        if ev is None:
            raise ValueError("no such eval run")
        ev.status = "running"
        ev.job_id = job["id"]
        mv = s.get(ModelVersion, ev.model_version_id)
        ds = s.get(Dataset, ev.dataset_id)
        if mv is None or ds is None:
            raise ValueError("eval run points at a missing model or dataset")
        snapshot = {"eval_id": ev.id, "team_id": ev.team_id, "mv_id": mv.id,
                    "base_model": mv.base_model, "adapter": mv.artifact_dir,
                    "dataset_path": ds.path, "name": f"{mv.name} v{mv.version}"}

    out = eval_dir(snapshot["eval_id"])
    ctx = _ctx_for(job, out)
    rows = load_rows(snapshot["dataset_path"])
    if not rows:
        raise ValueError("benchmark dataset is empty")

    backend = resolve_backend("auto", snapshot["base_model"])
    ctx.log(f"evaluating {snapshot['name']} on {len(rows)} rows (backend={backend})")

    from .trainers.factory import build_policy
    from .trainers.policy import build_batch, generate, sequence_logprob

    stub = TrainRequest(run_id=0, method="eval", base_model=snapshot["base_model"],
                        params={}, lora={}, out_dir=out, train_rows=rows, val_rows=[])
    policy = build_policy(stub, backend, adapter_dir=snapshot["adapter"], inference_only=True)

    def gen(prompts: list[str]) -> list[str]:
        return [o[0] for o in generate(policy, prompts, max_new_tokens=128, temperature=0.0)]

    def rank(prompt: str, choices: list[str]) -> int:
        batch = build_batch(policy.tokenizer, policy.device, [prompt] * len(choices),
                            choices, 1024)
        # Length-normalised: raw summed log-probability always prefers the
        # shortest option, which turns multiple choice into a length contest.
        lp = sequence_logprob(policy.model, batch, requires_grad=False, average=True)
        return int(lp.argmax())

    judge = get_judge() if any(r.get("rubric") for r in rows) else None
    metrics, per_example = evaluate_rows(
        rows, generate_fn=gen, rank_fn=rank, judge=judge,
        progress=lambda done, total: ctx.metric(done, total, evaluated=done),
    )
    metrics["backend"] = backend
    path = ctx.write_jsonl("per_example.jsonl", per_example)

    with get_session() as s:
        ev = s.get(EvalRun, snapshot["eval_id"])
        ev.status = "succeeded"
        ev.metrics = metrics
        ev.per_example_path = str(path)
        mv = s.get(ModelVersion, snapshot["mv_id"])
        if mv:
            mv.metrics = {**(mv.metrics or {}), **metrics}
            if mv.status == "draft":
                mv.status = "evaluated"
    return metrics


# ---------------------------------------------------------------- job: generate

def handle_generate(job: dict) -> dict[str, Any]:
    """Sample candidate responses and fill a review batch with them."""
    batch_id = job["payload"]["batch_id"]
    with get_session() as s:
        batch = s.get(ReviewBatch, batch_id)
        if batch is None:
            raise ValueError("no such review batch")
        batch.status = "generating"
        ds = s.get(Dataset, batch.prompt_dataset_id) if batch.prompt_dataset_id else None
        mv = s.get(ModelVersion, batch.policy_model_version_id) \
            if batch.policy_model_version_id else None
        snapshot = {
            "team_id": batch.team_id, "k": batch.candidates_per_prompt,
            "dataset_path": ds.path if ds else None,
            "base_model": mv.base_model if mv else None,
            "adapter": mv.artifact_dir if mv else None,
            "policy_name": f"{mv.name} v{mv.version}" if mv else None,
            "limit": int(job["payload"].get("limit", 100)),
        }

    if not snapshot["dataset_path"]:
        raise ValueError("review batch has no prompt dataset")
    rows = load_rows(snapshot["dataset_path"], limit=snapshot["limit"])
    if not rows:
        raise ValueError("prompt dataset is empty")

    out = run_dir(0) / f"batch_{batch_id}"
    ctx = _ctx_for(job, out)

    if not snapshot["base_model"]:
        raise ValueError("review batch has no policy model to sample from")
    backend = resolve_backend("auto", snapshot["base_model"])
    ctx.log(f"sampling {snapshot['k']} candidates × {len(rows)} prompts "
            f"from {snapshot['policy_name']} (backend={backend})")

    from .trainers.factory import build_policy, prompt_texts
    from .trainers.policy import generate

    stub = TrainRequest(run_id=0, method="generate", base_model=snapshot["base_model"],
                        params={}, lora={}, out_dir=out, train_rows=rows, val_rows=[])
    policy = build_policy(stub, backend, adapter_dir=snapshot["adapter"], inference_only=True)
    prompts = prompt_texts(policy, rows)

    created = 0
    for i in range(0, len(prompts), 8):
        ctx.check_cancelled()
        chunk = prompts[i:i + 8]
        # Temperature 1.0 on purpose: candidates that barely differ give a
        # reviewer nothing to choose between and produce a batch of forced ties.
        outs = generate(policy, chunk, max_new_tokens=160, temperature=1.0,
                        num_return_sequences=snapshot["k"])
        with get_session() as s:
            for prompt, cands in zip(chunk, outs):
                uniq = list(dict.fromkeys(c.strip() for c in cands if c.strip()))
                if len(uniq) < 2:
                    continue
                s.add(ReviewItem(
                    batch_id=batch_id, team_id=snapshot["team_id"], prompt=prompt,
                    candidates=[{"id": chr(97 + n), "text": t, "source": "policy"}
                                for n, t in enumerate(uniq[:snapshot["k"]])],
                    uncertainty=round(1.0 - _spread(uniq), 4),
                ))
                created += 1
        ctx.metric(min(i + 8, len(prompts)), len(prompts), items=created)

    with get_session() as s:
        batch = s.get(ReviewBatch, batch_id)
        batch.status = "open"
        followups = []
        if batch.ai_assist_fraction > 0:
            jjob = q.enqueue(s, team_id=batch.team_id, kind="judge",
                             payload={"batch_id": batch_id}, priority=4,
                             created_by=batch.created_by)
            followups.append({"judge_job": jjob.id})

    return {"items_created": created, "followups": followups}


def _spread(candidates: list[str]) -> float:
    """Rough lexical diversity across candidates, 0..1.

    Used as the inverse of ``uncertainty`` so the review queue prefers prompts
    where the policy produced genuinely different answers — those are the ones a
    human judgement actually resolves."""
    import re
    sets = [set(re.findall(r"[a-z0-9']+", c.lower())) for c in candidates]
    if len(sets) < 2:
        return 1.0
    sims = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            sims.append(len(sets[i] & sets[j]) / len(union) if union else 1.0)
    return sum(sims) / len(sims)


# ------------------------------------------------------------------- job: judge

def handle_judge(job: dict) -> dict[str, Any]:
    """Add an AI annotation to items in a batch."""
    batch_id = job["payload"]["batch_id"]
    with get_session() as s:
        batch = s.get(ReviewBatch, batch_id)
        if batch is None:
            raise ValueError("no such review batch")
        fraction = batch.ai_assist_fraction or 1.0
        model = batch.judge_model or judge_config()["model"]
        item_ids = list(s.execute(
            select(ReviewItem.id).where(ReviewItem.batch_id == batch_id)
        ).scalars())

    judge = get_judge(model)
    ctx = _ctx_for(job, run_dir(0) / f"judge_{batch_id}")
    take = max(1, int(len(item_ids) * fraction))
    item_ids = item_ids[:take]
    ctx.log(f"judging {len(item_ids)} items with {getattr(judge, 'model', model)}")

    judged = ties = flips = 0
    for n, item_id in enumerate(item_ids, 1):
        ctx.check_cancelled()
        with get_session() as s:
            item = s.get(ReviewItem, item_id)
            if item is None or len(item.candidates or []) < 2:
                continue
            a, b = item.candidates[0], item.candidates[1]
            try:
                if getattr(judge, "model", "") == "heuristic-v1":
                    v = judge.compare(item.prompt, a["text"], b["text"],
                                      reference=(item.meta or {}).get("reference"))
                else:
                    v = judge.compare(item.prompt, a["text"], b["text"])
            except Exception as e:
                ctx.log(f"judge error on item {item_id}: {e}", level="warn")
                continue

            choice = {"a": a["id"], "b": b["id"]}.get(v.winner, "tie")
            reviewlib.record_annotation(
                s, item=item, user=None, protocol="pairwise", choice=choice,
                rationale=v.rationale, confidence=v.confidence,
                annotator_type="ai", judge_model=v.judge_model,
            )
            judged += 1
            ties += int(choice == "tie")
            flips += int(v.position_flip)
        ctx.metric(n, len(item_ids), judged=judged, ties=ties, position_flips=flips)

    actual = getattr(judge, "model", model)
    if actual != model:
        # The batch asked for one judge and got another (no API key, most likely).
        # Record what actually ran — a batch row claiming Sonnet while every
        # annotation says heuristic-v1 is a stored contradiction.
        with get_session() as s:
            batch = s.get(ReviewBatch, batch_id)
            if batch:
                batch.judge_model = actual
        ctx.log(f"requested judge {model!r} was unavailable; used {actual!r}", level="warn")

    return {"judged": judged, "ties": ties, "position_flips": flips, "judge_model": actual}


# ---------------------------------------------------------------- job: assemble

def handle_assemble(job: dict) -> dict[str, Any]:
    batch_id = job["payload"]["batch_id"]
    with get_session() as s:
        stats = reviewlib.assemble_pairs(s, batch_id)
        batch = s.get(ReviewBatch, batch_id)
        report = reviewlib.agreement_report(s, batch.team_id, batch_id)
        append_audit(s, action="review.assembled", actor_id=batch.created_by,
                     actor_type="worker", team_id=batch.team_id,
                     subject_type="review_batch", subject_id=batch_id,
                     payload={**stats, "agreement": report.get("inter_annotator_agreement")})
    return {**stats, "agreement": report}


HANDLERS = {
    "train": handle_train,
    "eval": handle_eval,
    "generate": handle_generate,
    "judge": handle_judge,
    "assemble": handle_assemble,
}


# ---------------------------------------------------------------------- runloop

def _mark_run_failed(job: dict, message: str, terminal: bool) -> None:
    if not job.get("run_id"):
        return
    with get_session() as s:
        run = s.get(Run, job["run_id"])
        if run and terminal:
            run.status = "cancelled" if "cancel" in message.lower() else "failed"
            run.error = message[:8000]
            run.finished_at = utcnow()
        elif run:
            run.status = "queued"


def run_once(worker_id: str, kinds: list[str]) -> bool:
    job = q.claim(worker_id, kinds)
    if job is None:
        return False

    q.touch_worker(worker_id, job["id"])
    q.log(job["id"], f"starting {job['kind']}")
    try:
        result = HANDLERS[job["kind"]](job)
        q.finish(job["id"], job["lease_token"], status="succeeded",
                 progress={"pct": 100, "result": _trim(result)})
        q.log(job["id"], f"{job['kind']} complete", "info", **_trim(result))
    except Cancelled as e:
        q.finish(job["id"], job["lease_token"], status="cancelled", error=str(e))
        _mark_run_failed(job, str(e), terminal=True)
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        q.log(job["id"], traceback.format_exc()[-4000:], "error")
        status = q.fail_or_retry(job["id"], job["lease_token"], detail)
        _mark_run_failed(job, detail, terminal=status != "queued")
    finally:
        q.touch_worker(worker_id, None)
    return True


def _trim(result: Any) -> dict:
    if not isinstance(result, dict):
        return {"result": str(result)[:500]}
    return {k: v for k, v in result.items()
            if isinstance(v, (int, float, str, bool, list, dict))}


def serve(kinds: list[str], worker_id: str | None = None, poll_seconds: float = 2.0) -> None:
    worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
    init_db()
    capabilities = _capabilities()
    q.register_worker(worker_id, kinds, capabilities)
    print(f"[worker] {worker_id} kinds={','.join(kinds)} {capabilities}", flush=True)

    def _stop(signum, _frame):
        global _shutdown
        _shutdown = True
        print(f"[worker] signal {signum} — finishing current job then exiting", flush=True)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    last_reap = 0.0
    while not _shutdown:
        now = time.time()
        if now - last_reap > queue_config()["reap_interval_seconds"]:
            # Every worker reaps. It is idempotent, and a dedicated reaper process
            # would be one more thing whose own death goes unnoticed.
            requeued = q.reap_expired()
            if requeued:
                print(f"[worker] requeued {requeued} expired job(s)", flush=True)
            last_reap = now
            q.touch_worker(worker_id, None)

        if not run_once(worker_id, kinds):
            time.sleep(poll_seconds)

    print(f"[worker] {worker_id} stopped", flush=True)


def _capabilities() -> dict[str, Any]:
    caps: dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        import torch
        caps["torch"] = torch.__version__
        caps["cuda"] = torch.cuda.is_available()
        caps["mps"] = torch.backends.mps.is_available()
        if torch.cuda.is_available():
            caps["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        caps["torch"] = None
    from .config import anthropic_key
    caps["judge"] = "claude" if anthropic_key() else "heuristic-v1"
    return caps


def main() -> None:
    ap = argparse.ArgumentParser(description="SLM Foundry worker")
    ap.add_argument("--kinds", default=",".join(ALL_KINDS),
                    help=f"comma-separated job kinds ({'|'.join(ALL_KINDS)})")
    ap.add_argument("--worker-id", default=None)
    ap.add_argument("--once", action="store_true", help="run a single job and exit")
    ap.add_argument("--poll", type=float, default=2.0)
    args = ap.parse_args()

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    bad = [k for k in kinds if k not in ALL_KINDS]
    if bad:
        ap.error(f"unknown kinds: {bad}")

    if args.once:
        init_db()
        wid = args.worker_id or f"once-{uuid.uuid4().hex[:8]}"
        q.register_worker(wid, kinds, _capabilities())
        print("ran a job" if run_once(wid, kinds) else "queue empty")
        return
    serve(kinds, args.worker_id, args.poll)


if __name__ == "__main__":
    main()
