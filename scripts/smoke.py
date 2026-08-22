"""End-to-end smoke test: every pipeline, for real, in about a minute.

Runs on the ``tiny`` backend — a randomly-initialised two-layer model with a
vocabulary built from the seeded data — so no weights are downloaded and no GPU
is needed. The *training maths is not stubbed*: this exercises the real LoRA
injection, the real DPO log-ratio, the real PPO clipped surrogate with GAE, the
real GRPO group baseline and the real GSPO sequence ratio. What it proves is that
the plumbing is sound end to end; it proves nothing about model quality, and the
numbers it prints should never be quoted as results.

    PYTHONPATH=src python scripts/smoke.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TMP = Path(tempfile.mkdtemp(prefix="foundry-smoke-"))
os.environ["FOUNDRY_DATABASE_URL"] = f"sqlite:///{TMP}/smoke.db"
os.environ["FOUNDRY_DATA_DIR"] = str(TMP / "var")
os.environ["FOUNDRY_BACKEND"] = "tiny"
os.environ.pop("ANTHROPIC_API_KEY", None)      # exercise the offline judge path

from sqlalchemy import select                                        # noqa: E402

from foundry import queue as q                                       # noqa: E402
from foundry import review as reviewlib                              # noqa: E402
from foundry import worker                                           # noqa: E402
from foundry.config import defaults_for                              # noqa: E402
from foundry.db import get_session, init_db, verify_audit_chain      # noqa: E402
from foundry.models import (                                         # noqa: E402
    Dataset, EvalRun, ModelVersion, PreferencePair, ReviewBatch, ReviewItem, Run, Team, User,
)

PASS, FAIL = "\033[32m✓\033[0m", "\033[31m✗\033[0m"
results: list[tuple[str, bool, str]] = []


def check(label: str, fn):
    t0 = time.time()
    try:
        detail = fn() or ""
        results.append((label, True, f"{detail} ({time.time() - t0:.1f}s)"))
        print(f"{PASS} {label} — {detail} ({time.time() - t0:.1f}s)", flush=True)
        return True
    except Exception as e:
        results.append((label, False, f"{type(e).__name__}: {e}"))
        print(f"{FAIL} {label} — {type(e).__name__}: {e}", flush=True)
        traceback.print_exc(limit=4)
        return False


def drain(kinds=("train", "eval", "generate", "judge", "assemble"), limit: int = 30) -> int:
    """Run queued jobs inline until the queue is empty."""
    ran = 0
    while ran < limit and worker.run_once("smoke-worker", list(kinds)):
        ran += 1
    return ran


def submit(name: str, method: str, *, params=None, train_ds=None, eval_ds=None,
           parent=None, reward=None) -> int:
    with get_session() as s:
        team = s.execute(select(Team)).scalars().first()
        user = s.execute(select(User)).scalars().first()
        run = Run(team_id=team.id, name=name, method=method, base_model="tiny",
                  params=defaults_for(method) | (params or {}),
                  lora={"r": 4, "alpha": 8, "dropout": 0.0}, backend="tiny",
                  train_dataset_id=train_ds, eval_dataset_id=eval_ds,
                  parent_model_version_id=parent, reward_model_version_id=reward,
                  status="queued", created_by=user.id)
        s.add(run)
        s.flush()
        q.enqueue(s, team_id=team.id, kind="train", payload={"run_id": run.id},
                  run_id=run.id, created_by=user.id)
        return run.id


def finished(run_id: int) -> Run:
    with get_session() as s:
        run = s.get(Run, run_id)
        if run.status != "succeeded":
            raise AssertionError(f"run {run_id} ended {run.status}: {run.error[:400]}")
        return run


def model_for(run_id: int) -> ModelVersion:
    with get_session() as s:
        mv = s.execute(select(ModelVersion).where(ModelVersion.run_id == run_id)).scalar_one()
        return mv


def ds_id(kind: str) -> int:
    with get_session() as s:
        return s.execute(select(Dataset.id).where(Dataset.kind == kind)).scalars().first()


# --------------------------------------------------------------------------- run

print(f"\nSLM Foundry smoke test  ·  backend=tiny  ·  {TMP}\n" + "─" * 68)

init_db()
from seed import seed                                                # noqa: E402
seed()
print("─" * 68)

TINY = {"epochs": 1, "batch_size": 2, "max_seq_len": 128}
RL = {"iterations": 2, "rollout_batch": 2, "group_size": 4, "max_new_tokens": 16}

state: dict[str, int] = {}


def t_queue():
    with get_session() as s:
        team = s.execute(select(Team)).scalars().first()
        user = s.execute(select(User)).scalars().first()
        job = q.enqueue(s, team_id=team.id, kind="train", payload={"run_id": -1},
                        created_by=user.id, max_attempts=1)
        jid = job.id
    claimed = q.claim("w1", ["train"])
    assert claimed and claimed["id"] == jid, "claim did not return the queued job"
    assert q.claim("w2", ["train"]) is None, "a second worker claimed a leased job"
    assert q.heartbeat(jid, claimed["lease_token"], {"pct": 5}), "heartbeat rejected"
    assert not q.heartbeat(jid, "wrong-token", {}), "a stale token renewed the lease"
    q.finish(jid, claimed["lease_token"], status="cancelled")
    return "exclusive claim, lease enforcement, stale-token rejection"


def t_sft():
    rid = submit("smoke-sft", "sft", params=TINY, train_ds=ds_id("sft"),
                 eval_ds=ds_id("benchmark"))
    drain()
    run = finished(rid)
    mv = model_for(rid)
    state["sft_model"] = mv.id
    assert run.metrics["steps"] > 0
    assert "final_loss" in run.metrics
    return f"loss {run.metrics['final_loss']:.3f} over {run.metrics['steps']} steps → model #{mv.id}"


def t_eval():
    with get_session() as s:
        ev = s.execute(select(EvalRun).where(
            EvalRun.model_version_id == state["sft_model"])).scalar_one()
        assert ev.status == "succeeded", f"eval ended {ev.status}"
        assert ev.metrics.get("headline"), "no headline metric"
        return (f"{ev.metrics['headline']['label']} "
                f"{ev.metrics['headline']['value']} on {ev.metrics['n']} rows")


def t_reward():
    rid = submit("smoke-rm", "reward", params={"epochs": 1, "batch_size": 2, "max_seq_len": 128},
                 train_ds=ds_id("preference"))
    drain()
    run = finished(rid)
    mv = model_for(rid)
    state["reward_model"] = mv.id
    assert mv.artifact_kind == "reward"
    assert (Path(mv.artifact_dir) / "reward_head.pt").exists(), "reward head not saved"
    return f"held-out pair accuracy {run.metrics.get('heldout_pair_acc', 0):.3f}"


def t_dpo():
    rid = submit("smoke-dpo", "dpo",
                 params={"epochs": 1, "batch_size": 2, "max_seq_len": 128, "beta": 0.1},
                 train_ds=ds_id("preference"), parent=state["sft_model"])
    drain()
    run = finished(rid)
    state["dpo_model"] = model_for(rid).id
    for key in ("final_reward_acc", "final_logp_chosen", "final_implicit_kl"):
        assert key in run.metrics, f"missing {key}"
    return (f"reward_acc {run.metrics['final_reward_acc']:.3f}, "
            f"held-out {run.metrics.get('heldout_pref_acc', 0):.3f}")


def t_dpo_variants():
    out = []
    for variant in ("ipo", "hinge"):
        rid = submit(f"smoke-dpo-{variant}", "dpo",
                     params={"epochs": 1, "batch_size": 2, "max_seq_len": 128,
                             "loss_variant": variant},
                     train_ds=ds_id("preference"), parent=state["sft_model"])
        drain()
        out.append(f"{variant} {finished(rid).metrics['final_loss']:.3f}")
    return "loss — " + ", ".join(out)


def t_grpo():
    rid = submit("smoke-grpo", "grpo", params=defaults_for("grpo") | RL,
                 train_ds=ds_id("prompts"), parent=state["sft_model"])
    drain()
    run = finished(rid)
    assert "final_reward" in run.metrics and "final_kl" in run.metrics
    return (f"reward {run.metrics['final_reward']:.3f}, kl {run.metrics['final_kl']:.4f}, "
            f"{run.metrics['completions_sampled']} completions")


def t_gspo():
    rid = submit("smoke-gspo", "gspo", params=defaults_for("gspo") | RL | {"mu": 2},
                 train_ds=ds_id("prompts"), parent=state["sft_model"])
    drain()
    run = finished(rid)
    # The whole point of GSPO: a length-normalised sequence ratio that hugs 1.0
    # far more tightly than PPO's per-token ratios ever would.
    ratio = run.metrics["final_ratio"]
    assert 0.5 < ratio < 2.0, f"sequence ratio {ratio} is implausible"
    return f"sequence ratio {ratio:.5f}, clip_frac {run.metrics['final_clip_frac']:.3f}"


def t_ppo():
    rid = submit("smoke-ppo", "ppo",
                 params=defaults_for("ppo") | {"iterations": 2, "rollout_batch": 2,
                                               "ppo_epochs": 2, "max_new_tokens": 16},
                 train_ds=ds_id("prompts"), parent=state["sft_model"],
                 reward=state["reward_model"])
    drain()
    run = finished(rid)
    for key in ("final_value_loss", "final_explained_var", "final_kl"):
        assert key in run.metrics, f"missing {key}"
    return (f"value_loss {run.metrics['final_value_loss']:.3f}, "
            f"explained_var {run.metrics['final_explained_var']:.3f}, "
            f"kl_coef {run.metrics['final_kl_coef']}")


def t_rlaif():
    rid = submit("smoke-rlaif", "rlaif",
                 params=defaults_for("rlaif") | {"candidates_per_prompt": 2,
                                                 "max_new_tokens": 16, "min_margin": 0.0,
                                                 "human_spot_check": 0.5},
                 train_ds=ds_id("prompts"), parent=state["sft_model"])
    drain()
    run = finished(rid)
    assert run.metrics["feedback_source"] == "ai"
    assert run.metrics["judge_model"] == "heuristic-v1", "expected the offline judge"
    with get_session() as s:
        spot = s.execute(select(ReviewBatch).where(
            ReviewBatch.name.like("RLAIF spot-check%"))).scalars().first()
        assert spot is not None, "no human spot-check batch was queued"
        n = s.execute(select(ReviewItem).where(ReviewItem.batch_id == spot.id)).scalars().all()
    return (f"{run.metrics['pairs_generated']} pairs via {run.metrics['rlaif_optimizer'].upper()}, "
            f"{len(n)} queued for human spot-check")


def t_rlaif_grpo():
    rid = submit("smoke-rlaif-grpo", "rlaif",
                 params=defaults_for("rlaif") | {"optimizer": "grpo",
                                                 "candidates_per_prompt": 4,
                                                 "max_new_tokens": 16},
                 train_ds=ds_id("prompts"), parent=state["sft_model"])
    drain()
    run = finished(rid)
    assert run.metrics["rlaif_optimizer"] == "grpo"
    return f"AI feedback → GRPO, reward {run.metrics['final_reward']:.3f}"


def t_review_flow():
    with get_session() as s:
        team = s.execute(select(Team)).scalars().first()
        creator = s.execute(select(User).where(User.role == "member")).scalars().first()
        batch = ReviewBatch(team_id=team.id, name="smoke-batch", protocol="pairwise",
                            policy_model_version_id=state["sft_model"],
                            prompt_dataset_id=ds_id("prompts"), candidates_per_prompt=2,
                            annotations_per_item=2, ai_assist_fraction=1.0,
                            judge_model="heuristic-v1", status="draft", created_by=creator.id)
        s.add(batch)
        s.flush()
        bid = batch.id
        q.enqueue(s, team_id=team.id, kind="generate",
                  payload={"batch_id": bid, "limit": 6}, created_by=creator.id)
    drain()

    with get_session() as s:
        items = s.execute(select(ReviewItem).where(ReviewItem.batch_id == bid)).scalars().all()
        assert items, "candidate generation produced no review items"
        reviewers = s.execute(select(User).where(User.role == "reviewer")).scalars().all()
        assert len(reviewers) >= 2, "need two reviewers to measure agreement"

        assigned = 0
        for reviewer in reviewers:
            for _ in range(len(items)):
                result = reviewlib.assign_next(s, reviewer, bid)
                if result is None:
                    break
                item, prob = result
                assert 0 < prob <= 1, f"implausible sampling propensity {prob}"
                cands = item.candidates
                # Reviewer 1 always prefers the longer answer, reviewer 2 the first
                # listed: deliberately imperfect agreement, so the disputed path,
                # the consensus margin and the kappa all get exercised.
                choice = (max(cands, key=lambda c: len(c["text"]))["id"]
                          if reviewer.role == "reviewer" and reviewer.id % 2 else cands[0]["id"])
                reviewlib.record_annotation(s, item=item, user=reviewer, protocol="pairwise",
                                            choice=choice, confidence=0.9, latency_ms=4200)
                assigned += 1

        team = s.execute(select(Team)).scalars().first()
        q.enqueue(s, team_id=team.id, kind="assemble", payload={"batch_id": bid})
    drain()

    with get_session() as s:
        pairs = s.execute(select(PreferencePair).where(
            PreferencePair.batch_id == bid)).scalars().all()
        report = reviewlib.agreement_report(s, pairs[0].team_id if pairs else 1, bid)
        assert pairs, "no preference pairs assembled from the review batch"
        assert all(0 < p.weight <= 5 for p in pairs), "propensity weights out of range"
    return (f"{len(items)} items, {assigned} annotations, {len(pairs)} pairs; "
            f"IAA {report['inter_annotator_agreement']}, "
            f"human-vs-AI kappa {report['human_vs_ai_kappa']}")


def t_dpo_from_review():
    """The flywheel closing: review output feeds a training run with no dataset."""
    rid = submit("smoke-dpo-review", "dpo",
                 params={"epochs": 1, "batch_size": 2, "max_seq_len": 128},
                 parent=state["sft_model"])
    drain()
    run = finished(rid)
    return f"trained on review-collected pairs, loss {run.metrics['final_loss']:.3f}"


def t_promotion():
    from foundry import registry
    with get_session() as s:
        mv = s.get(ModelVersion, state["sft_model"])
        user = s.execute(select(User).where(User.role == "lead")).scalars().first()
        registry.promote(s, mv, to="staging", actor_id=user.id)
        registry.promote(s, mv, to="production", actor_id=user.id)
        assert mv.status == "production"
        assert Path(mv.artifact_dir).exists(), "promotion did not snapshot the artifact"

        # A second version of the same name must displace the first atomically.
        rival = ModelVersion(team_id=mv.team_id, name=mv.name, version=mv.version + 1,
                             base_model=mv.base_model, method="dpo",
                             artifact_dir=mv.artifact_dir, status="evaluated")
        s.add(rival)
        s.flush()
        registry.promote(s, rival, to="production", actor_id=user.id)
        s.refresh(mv)
        assert mv.status == "archived", "the incumbent was not demoted"
        return f"promoted, snapshotted, and displaced by v{rival.version}"


def t_tenancy():
    with get_session() as s:
        other = s.execute(select(User).where(User.email == "other@foundry.local")).scalar_one()
        mine = s.execute(select(Run)).scalars().first()
        assert mine.team_id != other.team_id, "fixture teams are not distinct"
        visible = s.execute(select(Run).where(Run.team_id == other.team_id)).scalars().all()
        assert not visible, "a neighbouring team can see this team's runs"
    return "cross-team queries return nothing"


def t_failure_path():
    with get_session() as s:
        team = s.execute(select(Team)).scalars().first()
        run = Run(team_id=team.id, name="smoke-broken", method="sft", base_model="tiny",
                  params=defaults_for("sft"), lora={}, backend="tiny", status="queued")
        s.add(run)
        s.flush()
        rid = run.id
        q.enqueue(s, team_id=team.id, kind="train", payload={"run_id": rid},
                  run_id=rid, max_attempts=1)
    drain()
    with get_session() as s:
        run = s.get(Run, rid)
        assert run.status == "failed", f"a run with no data ended {run.status}"
        assert "no training data" in run.error, run.error[:200]
    return "a run with no data fails cleanly with an actionable message"


def t_audit():
    with get_session() as s:
        result = verify_audit_chain(s)
        assert result["ok"], f"audit chain broken at seq {result.get('broken_at_seq')}"
        return f"{result['checked']} entries verified"


for label, fn in [
    ("queue: exclusive claim + leases", t_queue),
    ("SFT (LoRA)", t_sft),
    ("benchmark evaluation", t_eval),
    ("reward model (Bradley-Terry)", t_reward),
    ("DPO", t_dpo),
    ("DPO variants (IPO, hinge)", t_dpo_variants),
    ("GRPO", t_grpo),
    ("GSPO", t_gspo),
    ("PPO", t_ppo),
    ("RLAIF → DPO", t_rlaif),
    ("RLAIF → GRPO", t_rlaif_grpo),
    ("human review → preference pairs", t_review_flow),
    ("DPO from review data", t_dpo_from_review),
    ("registry promotion", t_promotion),
    ("tenancy isolation", t_tenancy),
    ("failure path", t_failure_path),
    ("audit chain", t_audit),
]:
    check(label, fn)

print("─" * 68)
passed = sum(1 for _, ok, _ in results if ok)
print(f"{passed}/{len(results)} passed   ·   artifacts in {TMP}")
if passed != len(results):
    print("\nfailures:")
    for label, ok, detail in results:
        if not ok:
            print(f"  {FAIL} {label}: {detail}")
sys.exit(0 if passed == len(results) else 1)
