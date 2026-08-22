"""Bradley-Terry reward model.

Given a pair (chosen, rejected) for the same prompt, the model is trained so that
``P(chosen ≻ rejected) = σ(r_chosen − r_rejected)`` matches the observed
preference. That is the whole objective; the useful part is what it produces —
a scalar scorer PPO can query on arbitrary generations, and an offline judge that
can rank a model's outputs without another round of human review.

Reward models are the component most likely to be quietly wrong, so two numbers
are logged on every step rather than at the end: **pair accuracy** (how often the
scorer agrees with the humans) and **reward margin** (how far apart it puts them).
Accuracy that stalls near 0.5 means the preferences carry no signal the model can
find; accuracy near 1.0 with a huge margin usually means it found a shortcut —
response length being the classic one.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .base import TrainContext, TrainRequest, TrainResult
from .factory import build_policy, describe, optimizer_for, prompt_texts, scheduler_for
from .policy import RewardModel, build_batch


def run(ctx: TrainContext, req: TrainRequest, backend: str) -> TrainResult:
    params = req.params
    rows = req.train_rows
    if not rows:
        raise ValueError("reward model training needs preference pairs")
    ctx.log(f"reward model: {len(rows)} pairs / {len(req.val_rows)} held out")

    policy = build_policy(req, backend)
    rm = RewardModel(policy)
    ctx.log("reward model ready", **describe(policy, backend))
    rm.train()

    prompts = prompt_texts(policy, rows)
    chosen = [r["chosen"] for r in rows]
    rejected = [r["rejected"] for r in rows]
    margins = [float(r.get("margin", 1.0)) for r in rows]

    bs = int(params["batch_size"])
    steps_per_epoch = max(1, math.ceil(len(rows) / bs))
    total_steps = max(1, int(steps_per_epoch * float(params["epochs"])))
    trainable = [p for p in rm.parameters() if p.requires_grad]
    opt = optimizer_for(trainable, float(params["learning_rate"]))
    sched = scheduler_for(opt, total_steps, 0.03)
    weighted = bool(params["margin_weighting"])

    g = torch.Generator().manual_seed(req.seed)
    order = list(range(len(rows)))

    for step in range(1, total_steps + 1):
        if (step - 1) % steps_per_epoch == 0:
            order = torch.randperm(len(rows), generator=g).tolist()
        start = ((step - 1) % steps_per_epoch) * bs
        idx = [order[i % len(order)] for i in range(start, start + bs)]

        # Chosen and rejected go through as one batch so the two sides share
        # padding width and batch-norm-free statistics — scoring them separately
        # makes the reward difference depend on batch composition.
        batch = build_batch(
            policy.tokenizer, policy.device,
            [prompts[i] for i in idx] * 2,
            [chosen[i] for i in idx] + [rejected[i] for i in idx],
            int(params["max_seq_len"]),
        )
        scores = rm(batch.input_ids, batch.attention_mask)
        r_chosen, r_rejected = scores[:len(idx)], scores[len(idx):]

        diff = r_chosen - r_rejected
        w = torch.tensor([margins[i] for i in idx], device=diff.device) if weighted \
            else torch.ones_like(diff)
        loss = -(F.logsigmoid(diff) * w).sum() / w.sum().clamp(min=1e-6)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()

        ctx.metric(step, total_steps,
                   loss=float(loss.detach()),
                   pair_acc=float((diff.detach() > 0).float().mean()),
                   reward_margin=float(diff.detach().mean()),
                   reward_chosen=float(r_chosen.detach().mean()),
                   reward_rejected=float(r_rejected.detach().mean()),
                   grad_norm=float(grad_norm),
                   lr=sched.get_last_lr()[0])

    val = _evaluate(rm, policy, req.val_rows, params)
    for k, v in val.items():
        ctx.log(f"held-out {k}: {v:.4f}")

    adapter_dir = ctx.out_dir / "reward_model"
    rm.save(adapter_dir, {"method": "reward", "run_id": req.run_id, "backend": backend})

    summary = ctx.summarise()
    summary.update({"backend": backend, "pairs": len(rows), **val})
    return TrainResult(metrics=summary, artifact_dir=str(adapter_dir),
                       artifact_kind="reward", backend=backend)


def _evaluate(rm: RewardModel, policy, rows: list[dict], params) -> dict[str, float]:
    if not rows:
        return {}
    prompts = prompt_texts(policy, rows)
    chosen = rm.score(prompts, [r["chosen"] for r in rows], int(params["max_seq_len"]))
    rejected = rm.score(prompts, [r["rejected"] for r in rows], int(params["max_seq_len"]))
    wins = sum(1 for a, b in zip(chosen, rejected) if a > b)
    diffs = [a - b for a, b in zip(chosen, rejected)]

    # Length correlation is the reward model's most common failure mode, so it is
    # reported unprompted rather than left for someone to discover in production.
    lens = [len(r["chosen"]) - len(r["rejected"]) for r in rows]
    corr = _pearson(diffs, [float(x) for x in lens])
    return {
        "heldout_pair_acc": wins / len(rows),
        "heldout_margin": sum(diffs) / len(diffs),
        "length_bias_corr": corr,
    }


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da and db else 0.0
