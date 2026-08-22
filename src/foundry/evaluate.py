"""Benchmarking: score a model version against a held-out dataset.

Whatever the benchmark rows carry, they get scored appropriately:

* ``choices`` + ``answer``  → **accuracy by likelihood ranking**. The model scores
  each option and the highest wins. This is deliberately not "generate and
  string-match the answer": a small model that knows the answer but phrases it
  loosely would be marked wrong, which measures formatting compliance rather than
  knowledge.
* ``reference``             → **exact match** and **token-F1** on a generation.
* ``rubric`` or nothing     → **judge score**, when a judge is configured.

Every run also reports generation length and a degeneracy rate, because the two
most common ways a fine-tune goes wrong — collapsing to one phrase, or never
stopping — both leave the task metric looking unremarkable.

A/B comparison is separate and pairwise: absolute scores across two runs drift
with prompt formatting and sampling settings, while "which of these two answers
is better" is the question a stakeholder actually asks before promoting a model.
"""
from __future__ import annotations

import math
import re
from typing import Any, Callable

from .rewards import token_f1
from .datasets import render_prompt

_WORD = re.compile(r"[a-z0-9']+")


def _norm(s: str) -> str:
    return " ".join(_WORD.findall(s.lower()))


def _degenerate(text: str) -> bool:
    """Collapsed output: one token repeated, or the same short phrase looping."""
    toks = _WORD.findall(text.lower())
    if len(toks) < 8:
        return False
    if len(set(toks)) / len(toks) < 0.25:
        return True
    for span in (2, 3, 4):
        grams = [tuple(toks[i:i + span]) for i in range(len(toks) - span)]
        if grams and max(grams.count(g) for g in set(grams)) > len(grams) * 0.4:
            return True
    return False


def evaluate_rows(
    rows: list[dict],
    *,
    generate_fn: Callable[[list[str]], list[str]],
    rank_fn: Callable[[str, list[str]], int] | None = None,
    judge=None,
    progress: Callable[[int, int], None] | None = None,
    batch_size: int = 8,
) -> tuple[dict[str, Any], list[dict]]:
    """Score every row. Returns (aggregate metrics, per-example records)."""
    mcq = [r for r in rows if r.get("choices")]
    gen = [r for r in rows if not r.get("choices")]
    per_example: list[dict] = []
    done = 0

    # ---- multiple choice, by likelihood -------------------------------------
    mcq_correct = 0
    if mcq and rank_fn is not None:
        for row in mcq:
            prompt = render_prompt(row["messages"])
            choices = [str(c) for c in row["choices"]]
            pick = rank_fn(prompt, choices)
            chosen = choices[pick]
            answer = row.get("answer")
            gold = choices[int(answer)] if str(answer).isdigit() and int(answer) < len(choices) \
                else str(answer)
            correct = _norm(chosen) == _norm(gold)
            mcq_correct += int(correct)
            per_example.append({"prompt": prompt, "kind": "choice", "predicted": chosen,
                                "gold": gold, "correct": correct})
            done += 1
            if progress:
                progress(done, len(rows))

    # ---- open generation -----------------------------------------------------
    exacts, f1s, lengths, degenerates, judged = [], [], [], 0, []
    for i in range(0, len(gen), batch_size):
        chunk = gen[i:i + batch_size]
        prompts = [render_prompt(r["messages"]) for r in chunk]
        outputs = generate_fn(prompts)
        for row, prompt, out in zip(chunk, prompts, outputs):
            rec: dict[str, Any] = {"prompt": prompt, "kind": "generation", "generated": out}
            reference = row.get("reference")
            if isinstance(reference, str) and reference.strip():
                rec["reference"] = reference
                rec["exact_match"] = _norm(out) == _norm(reference)
                rec["token_f1"] = round(token_f1(out, reference), 4)
                exacts.append(float(rec["exact_match"]))
                f1s.append(rec["token_f1"])
            if judge is not None:
                try:
                    rec["judge_score"] = round(_judge_score(judge, prompt, out, reference), 4)
                    judged.append(rec["judge_score"])
                except Exception as e:
                    rec["judge_error"] = str(e)[:200]
            rec["chars"] = len(out)
            rec["degenerate"] = _degenerate(out)
            lengths.append(len(out))
            degenerates += int(rec["degenerate"])
            per_example.append(rec)
            done += 1
        if progress:
            progress(done, len(rows))

    metrics: dict[str, Any] = {"n": len(rows), "n_choice": len(mcq), "n_generation": len(gen)}
    if mcq and rank_fn is not None:
        metrics["choice_accuracy"] = round(mcq_correct / len(mcq), 4)
    if exacts:
        metrics["exact_match"] = round(sum(exacts) / len(exacts), 4)
    if f1s:
        metrics["token_f1"] = round(sum(f1s) / len(f1s), 4)
    if judged:
        metrics["judge_score"] = round(sum(judged) / len(judged), 4)
    if lengths:
        metrics["mean_chars"] = round(sum(lengths) / len(lengths), 1)
        metrics["degenerate_rate"] = round(degenerates / len(lengths), 4)

    metrics["headline"] = _headline(metrics)
    return metrics, per_example


def _judge_score(judge, prompt: str, output: str, reference: str | None) -> float:
    if getattr(judge, "model", "") == "heuristic-v1":
        return judge.score(prompt, output, reference)
    return judge.score(prompt, output)


def _headline(metrics: dict[str, Any]) -> dict[str, Any]:
    """One number for the registry table, and its name.

    The registry lists models side by side, and a table where each row's headline
    came from a different metric would be actively misleading — so the choice is
    made once, here, in a fixed order of preference.
    """
    for key, label in (("choice_accuracy", "Accuracy"),
                       ("exact_match", "Exact match"),
                       ("token_f1", "Token F1"),
                       ("judge_score", "Judge score")):
        if key in metrics:
            return {"metric": key, "label": label, "value": metrics[key]}
    return {"metric": None, "label": "—", "value": None}


# ------------------------------------------------------------------ comparison

def compare_models(
    rows: list[dict],
    outputs_a: list[str],
    outputs_b: list[str],
    judge,
    *,
    debias: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Head-to-head win rate between two models' outputs on the same prompts."""
    wins_a = wins_b = ties = flips = 0
    records = []
    for i, (row, a, b) in enumerate(zip(rows, outputs_a, outputs_b), 1):
        prompt = render_prompt(row["messages"])
        try:
            if getattr(judge, "model", "") == "heuristic-v1":
                v = judge.compare(prompt, a, b, debias=debias, reference=row.get("reference"))
            else:
                v = judge.compare(prompt, a, b, debias=debias)
        except Exception:
            continue
        flips += int(v.position_flip)
        if v.winner == "a":
            wins_a += 1
        elif v.winner == "b":
            wins_b += 1
        else:
            ties += 1
        records.append({"prompt": prompt, "a": a, "b": b,
                        "winner": v.winner, "margin": round(v.margin, 3),
                        "rationale": v.rationale})
        if progress:
            progress(i, len(rows))

    decided = wins_a + wins_b
    win_rate = wins_b / decided if decided else None
    return {
        "n": len(records), "wins_a": wins_a, "wins_b": wins_b, "ties": ties,
        "win_rate_b": round(win_rate, 4) if win_rate is not None else None,
        "position_flip_rate": round(flips / max(len(records), 1), 4),
        # Without an interval, a 6-4 result on ten prompts reads as a win. With
        # it, it reads as the coin flip it is.
        "ci95": _wilson(wins_b, decided),
        "records": records[:50],
    }


def _wilson(successes: int, n: int, z: float = 1.96) -> list[float] | None:
    """Wilson score interval — correct near 0 and 1, where the normal
    approximation produces bounds outside [0,1]."""
    if n == 0:
        return None
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]
