"""RLAIF — reinforcement learning from AI feedback.

The important thing about RLAIF is what it does *not* replace. It substitutes the
labelling step, not the optimiser: a judge model stands in for the reviewers who
would otherwise sit in the review console, and the preferences it produces are
then consumed by DPO, GRPO or GSPO exactly as human preferences would be. That is
why ``optimizer`` is a first-class parameter here rather than a hidden default —
"we used RLAIF" says nothing about how the weights moved.

The pipeline:

    prompts → sample K candidates → judge ranks them → preference pairs → optimiser

Three safeguards, all on by default:

**Position debiasing.** The judge scores both orderings and the disagreement rate
is reported. A judge that flips on a third of its comparisons is a coin weighted
by presentation order, and the run says so rather than training on it.

**A minimum margin.** Pairs the judge could barely separate are discarded. A
forced verdict between two equivalent answers is noise with a label attached, and
it is worse than no data because it is indistinguishable from signal.

**A human spot-check.** A configurable fraction of judged pairs is also queued for
human review. This is the part that makes the judge falsifiable — without it,
RLAIF is a system whose label quality nobody has ever measured.
"""
from __future__ import annotations

import random

from ..judge import get_judge
from ..rewards import from_judge, references_from_rows
from .base import TrainContext, TrainRequest, TrainResult
from .factory import build_policy, describe, prompt_texts
from .policy import generate


def run(ctx: TrainContext, req: TrainRequest, backend: str) -> TrainResult:
    params = req.params
    rows = req.train_rows
    if not rows:
        raise ValueError("RLAIF needs a prompts dataset to sample from")

    optimizer = params["optimizer"]
    k = int(params["candidates_per_prompt"])
    debias = bool(params["position_debias"])
    min_margin = float(params["min_margin"]) / 5.0        # catalogue units are 0-5
    judge = get_judge(params["judge_model"])

    ctx.log(f"RLAIF: judge={getattr(judge, 'model', '?')} → optimiser={optimizer.upper()}, "
            f"K={k}, position_debias={debias}")
    if getattr(judge, "model", "") == "heuristic-v1":
        ctx.log("No ANTHROPIC_API_KEY configured — falling back to heuristic-v1, a mechanical "
                "scorer, not a model. Pairs from this run are labelled accordingly.", level="warn")

    # --- GRPO/GSPO consume the judge as a live reward function ------------------
    if optimizer in ("grpo", "gspo"):
        from .group_rl import run_group_rl

        policy = build_policy(req, backend)
        refs = references_from_rows(prompt_texts(policy, rows), rows)
        del policy                                          # group_rl builds its own

        inner = TrainRequest(
            run_id=req.run_id, method=optimizer, base_model=req.base_model,
            params=_merge_rl_params(optimizer, params), lora=req.lora,
            out_dir=req.out_dir, backend=req.backend,
            train_rows=rows, val_rows=req.val_rows,
            policy_adapter_dir=req.policy_adapter_dir,
            reward_fn=from_judge(judge, refs),
            seed=req.seed, extra=req.extra,
        )
        ctx.log("The judge is the reward function here, scoring each rollout directly. "
                "There are no chosen/rejected pairs to spot-check, so judge quality on this "
                "run is unmeasured — create a review batch on the same prompts if you need "
                "that number.", level="warn")
        result = run_group_rl(ctx, inner, backend, mode=optimizer)
        result.metrics.update({"feedback_source": "ai",
                               "judge_model": getattr(judge, "model", "unknown"),
                               "rlaif_optimizer": optimizer})
        return result

    # --- DPO consumes judged pairs --------------------------------------------
    policy = build_policy(req, backend)
    ctx.log("policy ready", **describe(policy, backend))
    prompts = prompt_texts(policy, rows)
    refs = references_from_rows(prompts, rows)

    ctx.log(f"sampling {k} candidates for each of {len(prompts)} prompts")
    batch_size = 8
    candidates: list[list[str]] = []
    for i in range(0, len(prompts), batch_size):
        ctx.check_cancelled()
        candidates.extend(generate(
            policy, prompts[i:i + batch_size],
            max_new_tokens=int(params["max_new_tokens"]),
            temperature=float(params["temperature"]),
            num_return_sequences=k,
        ))
        ctx.metric(min(i + batch_size, len(prompts)), len(prompts) * 2,
                   phase=1, sampled=min(i + batch_size, len(prompts)))

    pairs: list[dict] = []
    flips = ties = dropped = 0
    rng = random.Random(req.seed)

    for n, (prompt, outs) in enumerate(zip(prompts, candidates), 1):
        ctx.check_cancelled()
        uniq = list(dict.fromkeys(c for c in outs if c.strip()))
        if len(uniq) < 2:
            dropped += 1
            continue
        a, b = uniq[0], uniq[1]
        try:
            verdict = (judge.compare(prompt, a, b, debias=debias, reference=refs.get(prompt))
                       if getattr(judge, "model", "") == "heuristic-v1"
                       else judge.compare(prompt, a, b, debias=debias))
        except Exception as e:
            ctx.log(f"judge error on prompt {n}: {e}", level="warn")
            dropped += 1
            continue

        flips += int(verdict.position_flip)
        if verdict.winner == "tie" or verdict.margin < min_margin:
            ties += 1
            continue

        chosen, rejected = (a, b) if verdict.winner == "a" else (b, a)
        pairs.append({
            "messages": rows[n - 1].get("messages", []),
            "chosen": chosen, "rejected": rejected,
            "margin": round(verdict.margin, 4),
            "rationale": verdict.rationale,
            "judge_model": verdict.judge_model,
            "spot_check": rng.random() < float(params["human_spot_check"]),
        })
        ctx.metric(len(prompts) + n, len(prompts) * 2,
                   phase=2, pairs=len(pairs), ties=ties, position_flips=flips)

    flip_rate = flips / max(len(prompts), 1)
    ctx.log(f"judge produced {len(pairs)} usable pairs "
            f"({ties} ties/low-margin, {dropped} unusable, {flips} position flips)")
    if debias and flip_rate > 0.25:
        ctx.log(f"position-flip rate {flip_rate:.0%} — the judge is strongly order-dependent "
                f"on this task. Treat these labels as weak evidence.", level="warn")
    if not pairs:
        raise ValueError("the judge separated no pairs — lower min_margin, raise temperature, "
                         "or increase candidates_per_prompt")

    ctx.write_jsonl("judged_pairs.jsonl", pairs)
    del policy

    from . import dpo
    from ..config import defaults_for
    dpo_params = defaults_for("dpo") | {
        k2: v for k2, v in params.items() if k2 in defaults_for("dpo")
    }
    split = max(1, int(len(pairs) * 0.9))
    inner = TrainRequest(
        run_id=req.run_id, method="dpo", base_model=req.base_model,
        params=dpo_params, lora=req.lora, out_dir=req.out_dir, backend=req.backend,
        train_rows=pairs[:split], val_rows=pairs[split:],
        policy_adapter_dir=req.policy_adapter_dir, seed=req.seed, extra=req.extra,
    )
    result = dpo.run(ctx, inner, backend)
    result.metrics.update({
        "feedback_source": "ai",
        "judge_model": getattr(judge, "model", "unknown"),
        "rlaif_optimizer": "dpo",
        "pairs_generated": len(pairs),
        "pairs_tied": ties,
        "position_flip_rate": round(flip_rate, 4),
        "spot_check_queued": sum(1 for p in pairs if p["spot_check"]),
    })
    # The worker reads this to queue the spot-check sample into human review.
    result.samples = [p for p in pairs if p["spot_check"]]
    return result


def _merge_rl_params(optimizer: str, params: dict) -> dict:
    """Fill the chosen RL method's defaults, letting shared RLAIF keys win."""
    from ..config import defaults_for
    base = defaults_for(optimizer)
    for shared in ("temperature", "max_new_tokens"):
        if shared in params:
            base[shared] = params[shared]
    base["group_size"] = max(2, int(params["candidates_per_prompt"]))
    return base
