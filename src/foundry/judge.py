"""The RLAIF judge: an AI second opinion, recorded as one.

Two implementations behind one interface.

:class:`ClaudeJudge` calls the Anthropic API with a forced-schema tool so the
verdict comes back structured rather than parsed out of prose.

:class:`HeuristicJudge` is the no-API-key fallback. It is *not* a stand-in for a
model — it scores four stated, mechanical proxies (overlap with a reference,
degenerate repetition, length plausibility, and instruction echo) and is labelled
``heuristic-v1`` everywhere it appears, so nothing downstream can mistake its
output for a language model's judgement.

Three practices are baked in rather than left to the caller:

**Position debiasing.** Judges systematically favour whichever response they read
first. Every comparison is optionally run in both orders and averaged; the
residual disagreement between the two orderings is reported as ``position_flip``,
which is the cheapest available measure of how much to trust the judge at all.

**Ties are preserved.** A judge forced to break a tie invents a preference, and an
invented preference is indistinguishable from a real one downstream. ``tie`` is a
first-class verdict and the pair assembler drops it.

**The judge never sees the classifier's own prediction**, the policy's identity,
or which candidate came from the newer model. It sees two responses labelled A
and B.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from .config import anthropic_key, judge_config

SYSTEM_PROMPT = """You are an impartial evaluator comparing two candidate responses to the same prompt.

Judge only what is in front of you. You do not know which model produced which response, and there is no correct side. Weigh the criteria you are given, in the order given. Reward substance over length: a shorter response that fully answers is better than a longer one that circles.

Return a tie when the two are genuinely close. A forced preference between equivalent answers is noise that will be trained on."""

COMPARE_TOOL = {
    "name": "record_verdict",
    "description": "Record the comparison verdict and per-criterion scores.",
    "input_schema": {
        "type": "object",
        "properties": {
            "winner": {"type": "string", "enum": ["A", "B", "tie"],
                       "description": "Which response is better overall, or 'tie'."},
            "scores_a": {"type": "object", "description": "1-5 score per criterion key for A."},
            "scores_b": {"type": "object", "description": "1-5 score per criterion key for B."},
            "confidence": {"type": "number",
                           "description": "0-1. How clear the difference is."},
            "rationale": {"type": "string",
                          "description": "Two sentences maximum, naming the deciding difference."},
        },
        "required": ["winner", "scores_a", "scores_b", "confidence", "rationale"],
    },
}


@dataclass
class Verdict:
    winner: str                 # "a" | "b" | "tie"
    margin: float               # 0..1 strength of the preference
    confidence: float
    rationale: str
    scores: dict[str, Any]
    judge_model: str
    position_flip: bool = False


def _criteria_block(rubric: list[dict]) -> str:
    return "\n".join(f"- {c['label']} (key: {c['key']}, weight {c['weight']}): {c['guidance']}"
                     for c in rubric)


class ClaudeJudge:
    def __init__(self, model: str | None = None, rubric: list[dict] | None = None):
        cfg = judge_config()
        self.model = model or cfg["model"]
        self.rubric = rubric or cfg["rubric"]
        self.temperature = cfg.get("temperature", 0.0)
        from anthropic import Anthropic
        self.client = Anthropic(api_key=anthropic_key())

    def _once(self, prompt: str, a: str, b: str) -> dict[str, Any]:
        user = (
            f"# Criteria\n{_criteria_block(self.rubric)}\n\n"
            f"# Prompt\n{prompt}\n\n"
            f"# Response A\n{a}\n\n# Response B\n{b}\n\n"
            "Score each response 1-5 on every criterion key, then pick a winner."
        )
        msg = self.client.messages.create(
            model=self.model, max_tokens=1024, temperature=self.temperature,
            system=SYSTEM_PROMPT,
            tools=[COMPARE_TOOL], tool_choice={"type": "tool", "name": "record_verdict"},
            messages=[{"role": "user", "content": user}],
        )
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        raise RuntimeError("judge returned no verdict")

    def compare(self, prompt: str, a: str, b: str, *, debias: bool = True) -> Verdict:
        first = self._once(prompt, a, b)
        if not debias:
            return self._to_verdict(first, swapped=False)

        # Swap the presentation order; the judge's own position bias now points
        # the other way, so averaging cancels most of it.
        second = self._once(prompt, b, a)
        return self._merge(first, second)

    def _weighted(self, scores: dict) -> float:
        total, wsum = 0.0, 0.0
        for c in self.rubric:
            v = scores.get(c["key"])
            if isinstance(v, (int, float)):
                total += float(v) * c["weight"]
                wsum += c["weight"]
        return total / wsum if wsum else 0.0

    def _to_verdict(self, raw: dict, swapped: bool) -> Verdict:
        sa, sb = raw.get("scores_a", {}), raw.get("scores_b", {})
        if swapped:
            sa, sb = sb, sa
        winner = str(raw.get("winner", "tie")).lower()
        if swapped and winner in ("a", "b"):
            winner = "b" if winner == "a" else "a"
        gap = abs(self._weighted(sa) - self._weighted(sb))
        return Verdict(
            winner=winner if winner in ("a", "b") else "tie",
            margin=min(1.0, gap / 4.0),
            confidence=float(raw.get("confidence", 0.5)),
            rationale=str(raw.get("rationale", ""))[:2000],
            scores={"a": sa, "b": sb},
            judge_model=self.model,
        )

    def _merge(self, first: dict, second: dict) -> Verdict:
        v1 = self._to_verdict(first, swapped=False)
        v2 = self._to_verdict(second, swapped=True)
        flip = v1.winner != v2.winner
        if flip:
            # The two orderings disagree — that is a tie in everything but name,
            # and the margin is dropped to reflect it rather than picking a side.
            return Verdict(winner="tie", margin=0.0,
                           confidence=min(v1.confidence, v2.confidence) * 0.5,
                           rationale=f"Order-dependent verdict. A-first: {v1.rationale} "
                                     f"B-first: {v2.rationale}",
                           scores={"first": v1.scores, "swapped": v2.scores},
                           judge_model=self.model, position_flip=True)
        return Verdict(
            winner=v1.winner,
            margin=(v1.margin + v2.margin) / 2,
            confidence=(v1.confidence + v2.confidence) / 2,
            rationale=v1.rationale,
            scores={"first": v1.scores, "swapped": v2.scores},
            judge_model=self.model, position_flip=False,
        )

    def score(self, prompt: str, response: str) -> float:
        """Absolute 0-1 quality score, used as a reward signal by GRPO/GSPO."""
        user = (f"# Criteria\n{_criteria_block(self.rubric)}\n\n"
                f"# Prompt\n{prompt}\n\n# Response\n{response}\n\n"
                "Score this single response 1-5 on every criterion key. "
                "Put the same scores in scores_a and scores_b and answer 'tie'.")
        msg = self.client.messages.create(
            model=self.model, max_tokens=800, temperature=self.temperature,
            system=SYSTEM_PROMPT, tools=[COMPARE_TOOL],
            tool_choice={"type": "tool", "name": "record_verdict"},
            messages=[{"role": "user", "content": user}],
        )
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use":
                return self._weighted(dict(block.input).get("scores_a", {})) / 5.0
        return 0.0


# ------------------------------------------------------------- offline fallback

_WORD = re.compile(r"[a-z0-9']+")


def _tokens(s: str) -> list[str]:
    return _WORD.findall(s.lower())


class HeuristicJudge:
    """Mechanical scorer for when no API key is configured.

    Deliberately simple and deliberately named ``heuristic-v1`` in every record it
    writes. It measures four things it can actually measure — reference overlap,
    self-repetition, length plausibility, and prompt echo — and makes no claim
    beyond them. Its purpose is to keep the RLAIF pipeline runnable end to end,
    not to approximate a model's taste.
    """

    model = "heuristic-v1"

    def __init__(self, rubric: list[dict] | None = None):
        self.rubric = rubric or judge_config()["rubric"]

    def _components(self, prompt: str, response: str, reference: str | None) -> dict[str, float]:
        toks = _tokens(response)
        n = len(toks)
        if n == 0:
            return {"overlap": 0.0, "variety": 0.0, "length": 0.0, "echo": 0.0}

        variety = len(set(toks)) / n                      # 1.0 = no repetition

        ref_toks = set(_tokens(reference)) if reference else set()
        overlap = len(set(toks) & ref_toks) / max(len(ref_toks), 1) if ref_toks else 0.5

        # Plausibility peaks around 60 words and falls off either side.
        length = math.exp(-((math.log(max(n, 1) / 60)) ** 2) / 2)

        prompt_toks = set(_tokens(prompt))
        echo = 1.0 - (len(set(toks) & prompt_toks) / max(len(set(toks)), 1))

        return {"overlap": overlap, "variety": variety, "length": length, "echo": echo}

    def _score(self, prompt: str, response: str, reference: str | None = None) -> float:
        c = self._components(prompt, response, reference)
        return (0.40 * c["overlap"] + 0.25 * c["variety"]
                + 0.20 * c["length"] + 0.15 * c["echo"])

    def score(self, prompt: str, response: str, reference: str | None = None) -> float:
        return self._score(prompt, response, reference)

    def compare(self, prompt: str, a: str, b: str, *, debias: bool = True,
                reference: str | None = None) -> Verdict:
        sa = self._score(prompt, a, reference)
        sb = self._score(prompt, b, reference)
        gap = abs(sa - sb)
        winner = "tie" if gap < 0.02 else ("a" if sa > sb else "b")
        return Verdict(
            winner=winner, margin=min(1.0, gap * 4), confidence=min(1.0, gap * 6),
            rationale=(f"heuristic-v1: reference overlap, repetition, length plausibility "
                       f"and prompt echo score A={sa:.3f} vs B={sb:.3f}"),
            scores={"a": self._components(prompt, a, reference),
                    "b": self._components(prompt, b, reference)},
            judge_model=self.model,
        )


def get_judge(model: str | None = None, rubric: list[dict] | None = None):
    """Claude when a key is configured, the heuristic otherwise. Never raises —
    an RLAIF run should degrade to a weaker labeller with a loud name, not die."""
    if anthropic_key():
        try:
            return ClaudeJudge(model, rubric)
        except Exception:
            pass
    return HeuristicJudge(rubric)
