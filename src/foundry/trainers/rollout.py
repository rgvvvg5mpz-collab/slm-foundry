"""Sampling and reward assignment for the on-policy methods.

PPO, GRPO and GSPO differ in how they turn rewards into a gradient; they agree
completely on how to *get* the rewards. That shared half lives here.

The one subtlety worth stating: ``old_logprobs`` are recomputed with a forward
pass after generation rather than read out of the sampler. Decoding to text and
re-tokenizing does not always round-trip, and a single token of drift between the
sequence that was sampled and the sequence that is scored corrupts every
importance ratio in the batch — quietly, because the loss still looks fine.
Recomputing from the tokenized batch the trainer will actually use removes the
class of bug entirely.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from .policy import Batch, build_batch, generate, token_logprobs


@dataclass
class Rollout:
    prompts: list[str]              # one entry per sampled completion
    completions: list[str]
    rewards: list[float]
    group_ids: list[int]            # which prompt each completion came from
    batch: Batch | None = None
    old_logprobs: torch.Tensor | None = None
    ref_logprobs: torch.Tensor | None = None
    stats: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.completions)


def sample(
    policy,
    prompt_texts: list[str],
    *,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    reward_fn: Callable[[list[str], list[str]], list[float]],
    max_seq_len: int = 1024,
) -> Rollout:
    grouped = generate(policy, prompt_texts, max_new_tokens=max_new_tokens,
                       temperature=temperature, num_return_sequences=group_size)

    prompts, completions, group_ids = [], [], []
    for gi, (p, outs) in enumerate(zip(prompt_texts, grouped)):
        for text in outs:
            prompts.append(p)
            # An empty sample is a real event (the model emitted EOS immediately)
            # and must stay in the group: dropping it hides the failure from the
            # group baseline and makes the advantage look better than it is.
            completions.append(text if text.strip() else " ")
            group_ids.append(gi)

    rewards = [float(r) for r in reward_fn(prompts, completions)]

    lengths = [len(c) for c in completions]
    return Rollout(
        prompts=prompts, completions=completions, rewards=rewards, group_ids=group_ids,
        stats={
            "reward_mean": sum(rewards) / max(len(rewards), 1),
            "reward_min": min(rewards, default=0.0),
            "reward_max": max(rewards, default=0.0),
            "completion_chars_mean": sum(lengths) / max(len(lengths), 1),
            "distinct_completions": len(set(completions)) / max(len(completions), 1),
        },
    )


def score_rollout(policy, rollout: Rollout, max_seq_len: int) -> Rollout:
    """Attach the tokenized batch plus behaviour and reference log-probs."""
    from . import lora as loralib

    rollout.batch = build_batch(policy.tokenizer, policy.device,
                                rollout.prompts, rollout.completions, max_seq_len)
    with torch.no_grad():
        rollout.old_logprobs = token_logprobs(policy.model, rollout.batch,
                                              requires_grad=False).detach()
        with loralib.adapters_disabled(policy.model):
            rollout.ref_logprobs = token_logprobs(policy.model, rollout.batch,
                                                  requires_grad=False).detach()
    return rollout


def group_advantages(rewards: list[float], group_ids: list[int], *,
                     scale_by_std: bool, device) -> tuple[torch.Tensor, dict]:
    """Advantage = reward minus the group's own mean.

    This is the trick that lets GRPO and GSPO delete the critic: with G samples
    from the same prompt, the group mean is an unbiased, zero-extra-cost estimate
    of that prompt's value.

    ``scale_by_std`` divides by the group's standard deviation. It is standard
    GRPO and it stabilises early training, but it also up-weights prompts the
    policy already answers consistently — a prompt where every sample scores
    within a hair of the others gets its tiny differences amplified to unit
    scale. Turning it off (the "Dr. GRPO" correction) removes that bias.
    """
    by_group: dict[int, list[int]] = {}
    for i, g in enumerate(group_ids):
        by_group.setdefault(g, []).append(i)

    adv = [0.0] * len(rewards)
    degenerate = 0
    for members in by_group.values():
        vals = [rewards[i] for i in members]
        mean = sum(vals) / len(vals)
        if len(vals) > 1:
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            std = var ** 0.5
        else:
            std = 0.0
        if std < 1e-6:
            # Every sample in the group scored the same: there is nothing to
            # prefer, and dividing by ~0 would manufacture an enormous gradient
            # out of floating-point noise. Contribute zero instead.
            degenerate += 1
            for i in members:
                adv[i] = 0.0
            continue
        for i in members:
            a = rewards[i] - mean
            adv[i] = a / std if scale_by_std else a

    return torch.tensor(adv, dtype=torch.float32, device=device), {
        "degenerate_groups": degenerate,
        "groups": len(by_group),
        "advantage_absmean": sum(abs(a) for a in adv) / max(len(adv), 1),
    }
