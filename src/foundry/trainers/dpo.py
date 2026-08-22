"""DPO — Direct Preference Optimization.

The insight DPO trades on: the optimal policy for a KL-constrained reward
objective has a closed form, so a reward model is an unnecessary intermediary.
Rearranging it turns the whole of RLHF into a binary classification loss over
preference pairs, with the *implicit* reward being the policy's log-ratio against
a frozen reference:

    r(x, y) = β · log( π_θ(y|x) / π_ref(y|x) )
    L = −log σ( r(x, y_chosen) − r(x, y_rejected) )

No sampling, no rollout loop, no value function, no reward model to go stale.
That is why it is the default here and why anything else should be a considered
choice rather than a habit.

Two failure modes the metrics are chosen to expose:

**Both likelihoods collapsing.** DPO only constrains the *difference*, so it will
happily push the chosen response down as long as it pushes the rejected one down
harder. ``logp_chosen`` is logged every step for exactly this; if it falls
steadily, raise ``sft_weight`` (which adds an NLL term on the chosen response —
the RPO variant) or raise β.

**Silent saturation.** Once ``reward_acc`` pins at 1.0 the gradient has mostly
gone; more epochs past that point buy overfitting, not improvement.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .base import TrainContext, TrainRequest, TrainResult
from .factory import build_policy, describe, optimizer_for, prompt_texts, scheduler_for
from .policy import build_batch, sequence_logprob
from . import lora as loralib


def _pair_batch(policy, prompts, chosen, rejected, max_seq_len):
    """Chosen and rejected in one padded batch — identical padding on both sides."""
    return build_batch(policy.tokenizer, policy.device,
                       prompts * 2, chosen + rejected, max_seq_len)


def dpo_loss(policy_logps, ref_logps, *, beta, variant, label_smoothing, n_pairs,
             avg_policy_logps=None, avg_ref_logps=None):
    """Returns (loss, chosen_reward, rejected_reward, logits)."""
    pi_c, pi_r = policy_logps[:n_pairs], policy_logps[n_pairs:]
    ref_c, ref_r = ref_logps[:n_pairs], ref_logps[n_pairs:]

    chosen_reward = beta * (pi_c - ref_c).detach()
    rejected_reward = beta * (pi_r - ref_r).detach()
    logits = (pi_c - ref_c) - (pi_r - ref_r)

    if variant == "ipo":
        # IPO regresses the length-normalised log-ratio onto 1/(2β) instead of
        # driving it to infinity, which is what stops DPO from over-fitting a
        # deterministic preference into an arbitrarily confident policy.
        h = (avg_policy_logps[:n_pairs] - avg_ref_logps[:n_pairs]) - \
            (avg_policy_logps[n_pairs:] - avg_ref_logps[n_pairs:])
        loss = (h - 1 / (2 * beta)) ** 2
    elif variant == "hinge":
        loss = torch.relu(1 - beta * logits)
    else:                                                    # sigmoid — standard DPO
        scaled = beta * logits
        # Conservative DPO: assume `label_smoothing` of the labels are flipped, so
        # a confidently-wrong pair cannot dominate the gradient.
        loss = -F.logsigmoid(scaled) * (1 - label_smoothing) \
               - F.logsigmoid(-scaled) * label_smoothing

    return loss, chosen_reward, rejected_reward, logits


def run(ctx: TrainContext, req: TrainRequest, backend: str) -> TrainResult:
    params = req.params
    rows = req.train_rows
    if not rows:
        raise ValueError("DPO needs preference pairs — collect a review batch or upload them")
    ctx.log(f"DPO ({params['loss_variant']}): {len(rows)} pairs, beta={params['beta']}")

    policy = build_policy(req, backend)
    ctx.log("policy ready", **describe(policy, backend))
    if not loralib.lora_modules(policy.model):
        raise RuntimeError("DPO needs LoRA adapters to form the reference policy")
    policy.train()

    prompts = prompt_texts(policy, rows)
    chosen = [r["chosen"] for r in rows]
    rejected = [r["rejected"] for r in rows]
    weights = [float(r.get("margin", 1.0)) for r in rows]

    beta = float(params["beta"])
    variant = params["loss_variant"]
    smoothing = float(params["label_smoothing"])
    sft_weight = float(params["sft_weight"])
    max_len = int(params["max_seq_len"])
    bs = int(params["batch_size"])

    steps_per_epoch = max(1, math.ceil(len(rows) / bs))
    total_steps = max(1, int(steps_per_epoch * float(params["epochs"])))
    trainable = [p for p in policy.model.parameters() if p.requires_grad]
    opt = optimizer_for(trainable, float(params["learning_rate"]))
    sched = scheduler_for(opt, total_steps, 0.03)

    g = torch.Generator().manual_seed(req.seed)
    order = list(range(len(rows)))

    for step in range(1, total_steps + 1):
        if (step - 1) % steps_per_epoch == 0:
            order = torch.randperm(len(rows), generator=g).tolist()
        start = ((step - 1) % steps_per_epoch) * bs
        idx = [order[i % len(order)] for i in range(start, start + min(bs, len(order)))]

        batch = _pair_batch(policy, [prompts[i] for i in idx],
                            [chosen[i] for i in idx], [rejected[i] for i in idx], max_len)

        pi_logps = sequence_logprob(policy.model, batch, requires_grad=True)
        with loralib.adapters_disabled(policy.model), torch.no_grad():
            ref_logps = sequence_logprob(policy.model, batch, requires_grad=False)

        avg_pi = avg_ref = None
        if variant == "ipo":
            lengths = batch.completion_mask.sum(dim=-1).clamp(min=1)
            avg_pi, avg_ref = pi_logps / lengths, ref_logps / lengths

        n = len(idx)
        per_pair, r_chosen, r_rejected, logits = dpo_loss(
            pi_logps, ref_logps, beta=beta, variant=variant,
            label_smoothing=smoothing, n_pairs=n,
            avg_policy_logps=avg_pi, avg_ref_logps=avg_ref)

        # A 5-0 reviewer split should push harder than a 3-2 one; `margin` carries
        # that through from the review console into the gradient.
        w = torch.tensor([weights[i] for i in idx], device=per_pair.device)
        loss = (per_pair * w).sum() / w.sum().clamp(min=1e-6)

        nll = torch.tensor(0.0, device=loss.device)
        if sft_weight > 0:
            chosen_tokens = batch.completion_mask[:n].sum().clamp(min=1)
            nll = -pi_logps[:n].sum() / chosen_tokens
            loss = loss + sft_weight * nll

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()

        ctx.metric(
            step, total_steps,
            loss=float(loss.detach()),
            reward_acc=float((r_chosen > r_rejected).float().mean()),
            reward_margin=float((r_chosen - r_rejected).mean()),
            reward_chosen=float(r_chosen.mean()),
            reward_rejected=float(r_rejected.mean()),
            # Watch this: a steady fall means the policy is degrading the good
            # answers too, just more slowly than the bad ones.
            logp_chosen=float(pi_logps[:n].detach().mean()),
            logp_rejected=float(pi_logps[n:].detach().mean()),
            implicit_kl=float((pi_logps.detach() - ref_logps).abs().mean()),
            sft_nll=float(nll.detach()),
            grad_norm=float(grad_norm),
            lr=sched.get_last_lr()[0],
        )

    val = _evaluate(policy, req.val_rows, params, beta)
    for k, v in val.items():
        ctx.log(f"held-out {k}: {v:.4f}")

    adapter_dir = ctx.out_dir / "adapter"
    policy.save(adapter_dir, {"method": "dpo", "run_id": req.run_id, "backend": backend,
                              "beta": beta, "variant": variant})

    summary = ctx.summarise()
    summary.update({"backend": backend, "pairs": len(rows), **val})
    return TrainResult(metrics=summary, artifact_dir=str(adapter_dir),
                       artifact_kind="adapter", backend=backend)


def _evaluate(policy, rows, params, beta) -> dict[str, float]:
    """Held-out preference accuracy: does the tuned policy prefer what humans did?

    This is the number that actually says whether the run worked. Training-set
    reward accuracy always climbs; only this one generalises."""
    if not rows:
        return {}
    policy.eval()
    prompts = prompt_texts(policy, rows)
    max_len = int(params["max_seq_len"])
    bs = int(params["batch_size"])
    wins, total, margins = 0, 0, []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            sl = slice(i, i + bs)
            batch = _pair_batch(policy, prompts[sl], [r["chosen"] for r in rows[sl]],
                                [r["rejected"] for r in rows[sl]], max_len)
            pi = sequence_logprob(policy.model, batch, requires_grad=False)
            with loralib.adapters_disabled(policy.model):
                ref = sequence_logprob(policy.model, batch, requires_grad=False)
            n = len(rows[sl])
            d = beta * ((pi[:n] - ref[:n]) - (pi[n:] - ref[n:]))
            wins += int((d > 0).sum())
            total += n
            margins.extend(d.cpu().tolist())
    policy.train()
    return {
        "heldout_pref_acc": wins / max(total, 1),
        "heldout_margin": sum(margins) / max(len(margins), 1),
    }
