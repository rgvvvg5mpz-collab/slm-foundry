"""Where reward comes from.

Three sources, one signature — ``fn(prompts, completions) -> list[float]`` — so
PPO, GRPO and GSPO never learn which one they are using:

* **A trained reward model.** The classic RLHF path. Highest fidelity to what
  reviewers actually said, and the one that goes stale as the policy moves away
  from the distribution the scorer was fitted on.
* **A judge model.** RLAIF. No staleness, no separate training run, and a cost
  per rollout instead of a cost per campaign.
* **A verifier.** Where the prompts dataset carries a reference answer, reward is
  computed mechanically. Cheapest and completely ungameable, but only available
  for tasks with checkable answers.

Every source is normalised into roughly [0, 1] before it reaches an optimiser.
Not for tidiness: KL coefficients, clip ranges and advantage scales are all tuned
against an assumed reward magnitude, and swapping a source that emits logits in
[-8, 8] for one that emits [0, 1] silently changes the effective learning rate of
every RL hyperparameter in the catalogue.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Iterable

RewardFn = Callable[[list[str], list[str]], list[float]]

_WORD = re.compile(r"[a-z0-9']+")


def _tokens(s: str) -> list[str]:
    return _WORD.findall(s.lower())


def _squash(x: float, scale: float = 4.0) -> float:
    """Logistic squash into (0,1). Keeps raw reward-model logits from dominating
    the KL term purely by being large."""
    return 1.0 / (1.0 + math.exp(-x / scale))


# ------------------------------------------------------------- reward model

def from_reward_model(reward_model, max_seq_len: int = 1024) -> RewardFn:
    def fn(prompts: list[str], completions: list[str]) -> list[float]:
        raw = reward_model.score(prompts, completions, max_seq_len)
        return [_squash(r) for r in raw]
    return fn


# -------------------------------------------------------------------- judge

def from_judge(judge, references: dict[str, str] | None = None,
               cache: dict | None = None) -> RewardFn:
    """Score each completion with the judge.

    Cached on (prompt, completion): RL resamples the same prompt many times and
    duplicate generations are common early in training, when the policy is still
    close to greedy. Without the cache a GRPO run with G=8 pays for eight nearly
    identical judgements per prompt per iteration.
    """
    cache = cache if cache is not None else {}
    is_heuristic = getattr(judge, "model", "") == "heuristic-v1"

    def fn(prompts: list[str], completions: list[str]) -> list[float]:
        out: list[float] = []
        for p, c in zip(prompts, completions):
            key = (p, c)
            if key not in cache:
                try:
                    if is_heuristic:
                        cache[key] = judge.score(p, c, (references or {}).get(p))
                    else:
                        cache[key] = judge.score(p, c)
                except Exception:
                    # A judge outage must not poison the batch with a fake score.
                    # 0.5 is the neutral value: it contributes nothing to a
                    # group-relative advantage rather than pulling it somewhere.
                    cache[key] = 0.5
            out.append(float(cache[key]))
        return out
    return fn


# ----------------------------------------------------------------- verifier

def token_f1(pred: str, ref: str) -> float:
    p, r = _tokens(pred), _tokens(ref)
    if not p or not r:
        return 0.0
    common: dict[str, int] = {}
    rc: dict[str, int] = {}
    for t in r:
        rc[t] = rc.get(t, 0) + 1
    overlap = 0
    for t in p:
        if rc.get(t, 0) > common.get(t, 0):
            common[t] = common.get(t, 0) + 1
            overlap += 1
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def from_verifier(references: dict[str, str], *, exact_bonus: float = 0.3) -> RewardFn:
    """Mechanical reward against a known answer.

    Token-F1 with a bonus for an exact normalised match. The bonus matters: pure
    F1 gives a padded answer containing the right words nearly full credit, and a
    policy optimised against that learns to hedge with every plausible answer at
    once."""
    def fn(prompts: list[str], completions: list[str]) -> list[float]:
        out = []
        for p, c in zip(prompts, completions):
            ref = references.get(p)
            if not ref:
                out.append(0.5)
                continue
            f1 = token_f1(c, ref)
            exact = 1.0 if " ".join(_tokens(c)) == " ".join(_tokens(ref)) else 0.0
            out.append(min(1.0, f1 * (1 - exact_bonus) + exact * exact_bonus))
        return out
    return fn


# ---------------------------------------------------------------- composite

def combine(sources: Iterable[tuple[RewardFn, float]]) -> RewardFn:
    """Weighted blend, e.g. 0.7 × reward model + 0.3 × verifier.

    Useful when a task is partly checkable: the verifier anchors correctness while
    the reward model handles everything correctness does not cover."""
    items = list(sources)

    def fn(prompts: list[str], completions: list[str]) -> list[float]:
        total = [0.0] * len(prompts)
        wsum = sum(w for _, w in items) or 1.0
        for source, w in items:
            for i, v in enumerate(source(prompts, completions)):
                total[i] += w * v
        return [t / wsum for t in total]
    return fn


def references_from_rows(policy_prompts: list[str], rows: list[dict]) -> dict[str, str]:
    """Map rendered prompt text → reference answer, where the dataset has one."""
    refs: dict[str, str] = {}
    for prompt, row in zip(policy_prompts, rows):
        ref = row.get("reference") or row.get("completion") or (row.get("meta") or {}).get("reference")
        if isinstance(ref, str) and ref.strip():
            refs[prompt] = ref
    return refs
