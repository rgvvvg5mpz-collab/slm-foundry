"""GRPO and GSPO — critic-free group-relative policy optimisation.

Both sample a group of G completions per prompt and use the group's mean reward
as the baseline, so neither needs a value network. They differ in exactly one
place, and it is the place that matters:

**GRPO** forms the importance ratio *per token*::

    r_{i,t} = π_θ(y_{i,t}) / π_old(y_{i,t})
    L = −(1/Σ|y_i|) Σ_i Σ_t min( r_{i,t}·A_i , clip(r_{i,t}, 1±ε)·A_i )

**GSPO** forms it *per sequence*, length-normalised::

    s_i = ( π_θ(y_i) / π_old(y_i) )^(1/|y_i|)
        = exp( (1/|y_i|) Σ_t [ log π_θ(y_{i,t}) − log π_old(y_{i,t}) ] )
    L = −(1/N) Σ_i min( s_i·A_i , clip(s_i, 1−ε_low, 1+ε_high)·A_i )

Why that matters: the advantage A_i is a *sequence*-level quantity — one reward
for the whole completion. GRPO nonetheless applies a per-token correction to it,
so on a 500-token generation the update accumulates 500 independently noisy
ratios against a single reward signal. One token whose probability moved sharply
can dominate, and clipping cannot help because it clips each token separately.
GSPO matches the granularity of the correction to the granularity of the reward:
one ratio, one clip, one decision per sequence.

The length normalisation is why GSPO's clip ranges look absurd next to PPO's.
A geometric mean over hundreds of tokens sits extremely close to 1.0, so ε on the
order of 3e-4 is the *equivalent* strictness — copying PPO's 0.2 across would
disable clipping entirely.
"""
from __future__ import annotations

import torch

from . import rollout as rollib
from .base import TrainContext, TrainRequest, TrainResult
from .factory import build_policy, describe, optimizer_for, prompt_texts, scheduler_for
from .policy import kl_k3, token_logprobs


def run_group_rl(ctx: TrainContext, req: TrainRequest, backend: str, mode: str) -> TrainResult:
    assert mode in ("grpo", "gspo")
    params = req.params
    rows = req.train_rows
    if not rows:
        raise ValueError(f"{mode.upper()} needs a prompts dataset to sample from")
    if req.reward_fn is None:
        raise ValueError(f"{mode.upper()} needs a reward source (reward model, judge, or verifier)")

    policy = build_policy(req, backend)
    ctx.log(f"{mode.upper()} ready", **describe(policy, backend))
    policy.train()

    all_prompts = prompt_texts(policy, rows)
    iterations = int(params["iterations"])
    rollout_batch = min(int(params["rollout_batch"]), len(all_prompts))
    group_size = int(params["group_size"])
    mu = int(params["mu"])
    kl_coef = float(params["kl_coef"])
    scale_by_std = bool(params["scale_by_std"])
    max_seq_len = int(req.extra.get("max_seq_len", 1024))

    if mode == "grpo":
        clip_low = clip_high = float(params["clip_range"])
    else:
        clip_low, clip_high = float(params["clip_low"]), float(params["clip_high"])

    trainable = [p for p in policy.model.parameters() if p.requires_grad]
    opt = optimizer_for(trainable, float(params["learning_rate"]))
    sched = scheduler_for(opt, iterations * mu, 0.03)

    ctx.log(f"{iterations} iterations × {rollout_batch} prompts × G={group_size} "
            f"= {iterations * rollout_batch * group_size} completions, mu={mu}")
    if mode == "gspo" and mu == 1:
        ctx.log("mu=1 makes every sequence ratio exactly 1.0 — GSPO's off-policy "
                "tolerance only pays off with mu>1", level="warn")

    g = torch.Generator().manual_seed(req.seed)
    step = 0
    total_steps = iterations * mu
    best_reward = float("-inf")
    last_samples: list[dict] = []

    for it in range(1, iterations + 1):
        ctx.check_cancelled()
        idx = torch.randperm(len(all_prompts), generator=g)[:rollout_batch].tolist()
        batch_prompts = [all_prompts[i] for i in idx]

        ro = rollib.sample(
            policy, batch_prompts,
            group_size=group_size,
            max_new_tokens=int(params["max_new_tokens"]),
            temperature=float(params["temperature"]),
            reward_fn=req.reward_fn,
            max_seq_len=max_seq_len,
        )
        rollib.score_rollout(policy, ro, max_seq_len)
        advantages, adv_stats = rollib.group_advantages(
            ro.rewards, ro.group_ids, scale_by_std=scale_by_std, device=policy.device)

        best_reward = max(best_reward, ro.stats["reward_max"])
        mask = ro.batch.completion_mask
        lengths = mask.sum(dim=-1).clamp(min=1)

        for _ in range(mu):
            step += 1
            logp = token_logprobs(policy.model, ro.batch, requires_grad=True)

            if mode == "grpo":
                ratio = torch.exp(logp - ro.old_logprobs) * mask
                adv = advantages.unsqueeze(-1)
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1 - clip_low, 1 + clip_high) * adv
                objective = torch.min(unclipped, clipped) * mask
                pg_loss = -objective.sum() / mask.sum().clamp(min=1)
                clip_frac = float(((ratio.detach() - 1).abs() > clip_low).float().mul(mask).sum()
                                  / mask.sum().clamp(min=1))
                ratio_report = float((ratio.detach().sum() / mask.sum().clamp(min=1)))
            else:
                # Sequence ratio: the geometric mean of the per-token ratios.
                seq_delta = (logp - ro.old_logprobs).sum(dim=-1) / lengths
                s = torch.exp(seq_delta)
                unclipped = s * advantages
                clipped = torch.clamp(s, 1 - clip_low, 1 + clip_high) * advantages
                pg_loss = -torch.min(unclipped, clipped).mean()
                sd = s.detach()
                clip_frac = float(((sd < 1 - clip_low) | (sd > 1 + clip_high)).float().mean())
                ratio_report = float(s.detach().mean())

            kl = kl_k3(logp, ro.ref_logprobs, mask)
            kl_per_token = kl.sum() / mask.sum().clamp(min=1)
            loss = pg_loss + kl_coef * kl_per_token

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()

            ctx.metric(
                step, total_steps,
                iteration=it,
                loss=float(loss.detach()),
                pg_loss=float(pg_loss.detach()),
                reward=ro.stats["reward_mean"],
                reward_max=ro.stats["reward_max"],
                kl=float(kl_per_token.detach()),
                ratio=ratio_report,
                clip_frac=clip_frac,
                advantage_absmean=adv_stats["advantage_absmean"],
                degenerate_groups=adv_stats["degenerate_groups"],
                completion_len=float(lengths.float().mean()),
                distinct=ro.stats["distinct_completions"],
                grad_norm=float(grad_norm),
                lr=sched.get_last_lr()[0],
            )

        if mode == "gspo" and clip_frac > 0.9:
            ctx.log(f"iteration {it}: {clip_frac:.0%} of sequences hit the clip. The policy is "
                    f"moving far further per step than clip_low={clip_low} allows, so almost "
                    f"every update is being truncated — lower the learning rate or widen the "
                    f"clip range.", level="warn")
        if adv_stats["degenerate_groups"] == adv_stats["groups"]:
            ctx.log(f"iteration {it}: every group had identical rewards — no learning "
                    f"signal. Raise temperature or check the reward source.", level="warn")

        last_samples = [
            {"prompt": p, "generated": c, "reward": round(r, 4), "advantage": round(float(a), 4)}
            for p, c, r, a in list(zip(ro.prompts, ro.completions, ro.rewards,
                                       advantages.cpu().tolist()))[:8]
        ]

    ctx.write_jsonl("samples.jsonl", last_samples)
    adapter_dir = ctx.out_dir / "adapter"
    policy.save(adapter_dir, {"method": mode, "run_id": req.run_id, "backend": backend})

    summary = ctx.summarise()
    summary.update({"backend": backend, "iterations": iterations,
                    "group_size": group_size,
                    "completions_sampled": iterations * rollout_batch * group_size,
                    "best_reward": round(best_reward, 4)})
    return TrainResult(metrics=summary, artifact_dir=str(adapter_dir),
                       artifact_kind="adapter", samples=last_samples, backend=backend)
