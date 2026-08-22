"""PPO — the original RLHF loop, with a learned critic.

Per iteration: sample completions, score them with the reward model, convert the
scores into per-token returns, fit a value function, and take a few clipped
policy steps inside a trust region around the sampling policy.

Two details carry most of the correctness:

**Where the reward goes.** The reward model produces one scalar for a whole
completion, but the policy gradient needs a signal at every token. The standard
construction — used here — puts the KL penalty at every token and the scalar
reward at the final one::

    r_t = −β · KL_t                      for t < T
    r_T = −β · KL_T + reward_model(y)

The KL term is not regularisation-as-garnish. It is what stops the policy from
walking off to whatever degenerate string the reward model happens to score
highest, and it is why the KL trace deserves as much attention as the reward one.
A run whose reward climbs while KL explodes has not improved; it has found a bug
in its reward model.

**Adaptive KL control.** A fixed β is either too loose early or too tight late.
The coefficient here tracks a target KL and adjusts multiplicatively, so the
operator sets "how far may this drift" rather than tuning a penalty by feel. Set
``target_kl`` to 0 to hold β fixed.
"""
from __future__ import annotations

import torch

from . import rollout as rollib
from .base import TrainContext, TrainRequest, TrainResult
from .factory import build_policy, describe, optimizer_for, prompt_texts, scheduler_for
from .policy import ValueHead, kl_k3, masked_whiten, policy_forward_with_values


def compute_gae(rewards, values, mask, gamma: float, lam: float):
    """Generalised advantage estimation, backwards over the completion.

    Bootstrapping stops at the mask boundary rather than the tensor edge — padded
    positions carry no value and letting them into the recursion drags every real
    advantage toward zero by however much padding the batch happened to have.
    """
    T = rewards.shape[1]
    advantages = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[:, 0])
    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t + 1 < T else torch.zeros_like(values[:, 0])
        next_mask = mask[:, t + 1] if t + 1 < T else torch.zeros_like(mask[:, 0])
        delta = rewards[:, t] + gamma * next_value * next_mask - values[:, t]
        last = delta + gamma * lam * next_mask * last
        advantages[:, t] = last * mask[:, t]
    return advantages, (advantages + values) * mask


def run(ctx: TrainContext, req: TrainRequest, backend: str) -> TrainResult:
    params = req.params
    rows = req.train_rows
    if not rows:
        raise ValueError("PPO needs a prompts dataset to sample from")
    if req.reward_fn is None:
        raise ValueError("PPO needs a reward model — train one first, or pick GRPO/GSPO")

    policy = build_policy(req, backend)
    ctx.log("PPO ready", **describe(policy, backend))
    hidden = getattr(policy.model.config, "hidden_size", None) or policy.model.config.n_embd
    value_head = ValueHead(hidden).to(policy.device)
    policy.train()

    all_prompts = prompt_texts(policy, rows)
    iterations = int(params["iterations"])
    rollout_batch = min(int(params["rollout_batch"]), len(all_prompts))
    ppo_epochs = int(params["ppo_epochs"])
    clip = float(params["clip_range"])
    vclip = float(params["value_clip_range"])
    vf_coef = float(params["vf_coef"])
    ent_coef = float(params["entropy_coef"])
    gamma, lam = float(params["gamma"]), float(params["lam"])
    kl_coef = float(params["kl_coef"])
    target_kl = float(params["target_kl"])
    max_seq_len = int(req.extra.get("max_seq_len", 1024))

    trainable = [p for p in policy.model.parameters() if p.requires_grad] + \
                list(value_head.parameters())
    opt = optimizer_for(trainable, float(params["learning_rate"]))
    total_steps = iterations * ppo_epochs
    sched = scheduler_for(opt, total_steps, 0.03)

    ctx.log(f"{iterations} iterations × {rollout_batch} prompts, {ppo_epochs} PPO epochs each")

    g = torch.Generator().manual_seed(req.seed)
    step = 0
    last_samples: list[dict] = []

    for it in range(1, iterations + 1):
        ctx.check_cancelled()
        idx = torch.randperm(len(all_prompts), generator=g)[:rollout_batch].tolist()

        ro = rollib.sample(
            policy, [all_prompts[i] for i in idx],
            group_size=1,
            max_new_tokens=int(params["max_new_tokens"]),
            temperature=float(params["temperature"]),
            reward_fn=req.reward_fn, max_seq_len=max_seq_len,
        )
        rollib.score_rollout(policy, ro, max_seq_len)
        mask = ro.batch.completion_mask

        with torch.no_grad():
            _, old_values = policy_forward_with_values(policy.model, value_head, ro.batch)
            old_values = old_values.detach()

        # Per-token KL penalty, plus the scalar reward at the last real token.
        kl_tok = kl_k3(ro.old_logprobs, ro.ref_logprobs, mask)
        token_rewards = -kl_coef * kl_tok
        scores = torch.tensor(ro.rewards, dtype=torch.float32, device=policy.device)
        last_idx = mask.sum(dim=-1).long().clamp(min=1) - 1
        token_rewards[torch.arange(len(ro), device=policy.device), last_idx] += scores

        advantages, returns = compute_gae(token_rewards, old_values, mask, gamma, lam)
        if bool(params["whiten_advantages"]):
            advantages = masked_whiten(advantages, mask)

        for _ in range(ppo_epochs):
            step += 1
            logp, values = policy_forward_with_values(policy.model, value_head, ro.batch)

            ratio = torch.exp(logp - ro.old_logprobs) * mask
            pg_unclipped = ratio * advantages
            pg_clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
            pg_loss = -(torch.min(pg_unclipped, pg_clipped) * mask).sum() / mask.sum().clamp(min=1)

            # Clipping the value function too keeps the critic inside the same
            # trust region as the policy; an unconstrained critic can move far
            # enough in one epoch to invalidate the advantages computed from it.
            v_clipped = old_values + torch.clamp(values - old_values, -vclip, vclip)
            v_loss = 0.5 * torch.max((values - returns) ** 2, (v_clipped - returns) ** 2)
            v_loss = (v_loss * mask).sum() / mask.sum().clamp(min=1)

            entropy = -(logp * mask).sum() / mask.sum().clamp(min=1)
            loss = pg_loss + vf_coef * v_loss - ent_coef * entropy

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()

            with torch.no_grad():
                kl_now = float((kl_k3(logp, ro.ref_logprobs, mask).sum()
                                / mask.sum().clamp(min=1)))

            ctx.metric(
                step, total_steps,
                iteration=it,
                loss=float(loss.detach()), pg_loss=float(pg_loss.detach()),
                value_loss=float(v_loss.detach()),
                reward=ro.stats["reward_mean"], reward_max=ro.stats["reward_max"],
                kl=kl_now, kl_coef=kl_coef,
                clip_frac=float(((ratio.detach() - 1).abs() > clip).float().mul(mask).sum()
                                / mask.sum().clamp(min=1)),
                entropy=float(entropy.detach()),
                explained_var=_explained_variance(old_values, returns, mask),
                completion_len=float(mask.sum(dim=-1).float().mean()),
                grad_norm=float(grad_norm), lr=sched.get_last_lr()[0],
            )

        if target_kl > 0:
            # Multiplicative controller, clamped to ±20% per iteration: a single
            # bad batch should nudge the penalty, not swing it.
            error = kl_now / target_kl - 1
            kl_coef = float(max(1e-4, min(10.0, kl_coef * (1 + max(-0.2, min(0.2, error))))))

        last_samples = [
            {"prompt": p, "generated": c, "reward": round(r, 4)}
            for p, c, r in list(zip(ro.prompts, ro.completions, ro.rewards))[:8]
        ]

    ctx.write_jsonl("samples.jsonl", last_samples)
    adapter_dir = ctx.out_dir / "adapter"
    policy.save(adapter_dir, {"method": "ppo", "run_id": req.run_id, "backend": backend})
    torch.save(value_head.state_dict(), adapter_dir / "value_head.pt")

    summary = ctx.summarise()
    summary.update({"backend": backend, "iterations": iterations,
                    "final_kl_coef": round(kl_coef, 5),
                    "completions_sampled": iterations * rollout_batch})
    return TrainResult(metrics=summary, artifact_dir=str(adapter_dir),
                       artifact_kind="adapter", samples=last_samples, backend=backend)


def _explained_variance(values, returns, mask) -> float:
    """How much of the return variance the critic accounts for.

    The single most diagnostic PPO number. Near 0 means the value head is not
    learning and every advantage is mostly noise; negative means it is actively
    worse than predicting the mean, and the run is not doing what it appears to.
    """
    with torch.no_grad():
        n = mask.sum().clamp(min=1)
        r_mean = (returns * mask).sum() / n
        var_r = (((returns - r_mean) ** 2) * mask).sum() / n
        var_e = (((returns - values) ** 2) * mask).sum() / n
        if float(var_r) < 1e-8:
            return 0.0
        return float(1 - var_e / var_r)
