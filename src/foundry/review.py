"""The human review layer: assignment, judgement, consensus, and pair assembly.

This is the module the whole product turns on. Every preference method downstream
consumes exactly one thing — pairs of (chosen, rejected) for the same prompt —
and this is where those come from, whether the judgement was made by a person or
by a judge model.

Four decisions worth defending:

**Assignment is sampled, not sorted.** A strict "hardest item first" queue is a
deterministic policy with an unknown propensity, which makes every rate estimated
from the reviewed data biased by an unknown amount. Here the eligible pool is
scored, softmaxed, and *sampled*, and the resulting probability is stored on the
assignment. That single float is what lets a win-rate computed on reviewed items
mean anything about the population they were drawn from.

**A reviewer never sees the same item twice**, and never sees which candidate came
from the newer model. Candidate order is shuffled per assignment and the mapping
lives in the assignment, not the item.

**Ties are kept and then dropped.** A reviewer can say "tie" or "both bad", those
are recorded, and the pair assembler excludes them. Forcing a preference produces
labels that look exactly like real ones and are not.

**Consensus is computed, not assumed.** With multiple annotators per item, the
margin passed to DPO reflects how lopsided the vote was — a 5-0 split pushes
harder than a 3-2, which is the whole reason ``margin`` exists on the pair.
"""
from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import (
    Adjudication, Annotation, PreferencePair, ReviewAssignment, ReviewBatch,
    ReviewItem, User, utcnow,
)

TIE_CHOICES = ("tie", "both_bad")


# --------------------------------------------------------------------- assignment

def _needed(s: Session, batch: ReviewBatch, item_ids: list[int]) -> dict[int, int]:
    """How many more annotations each item still wants."""
    counts = dict(
        s.execute(
            select(Annotation.item_id, func.count())
            .where(Annotation.item_id.in_(item_ids),
                   Annotation.annotator_type == "human",
                   Annotation.superseded_by.is_(None))
            .group_by(Annotation.item_id)
        ).all()
    )
    return {i: max(0, batch.annotations_per_item - counts.get(i, 0)) for i in item_ids}


def assign_next(s: Session, user: User, batch_id: int | None = None,
                pool_size: int = 40, temperature: float = 0.5) -> tuple[ReviewItem, float] | None:
    """Hand this reviewer their next item, with the propensity that produced it."""
    already = {
        r for r in s.execute(
            select(ReviewAssignment.item_id).where(ReviewAssignment.user_id == user.id)
        ).scalars()
    }

    q = (select(ReviewItem)
         .join(ReviewBatch, ReviewBatch.id == ReviewItem.batch_id)
         .where(ReviewItem.team_id == user.team_id,
                ReviewItem.status.in_(("pending", "in_review")),
                ReviewBatch.status == "open"))
    if batch_id:
        q = q.where(ReviewItem.batch_id == batch_id)
    candidates = [i for i in s.execute(q.limit(pool_size * 4)).scalars() if i.id not in already]
    if not candidates:
        return None

    batches = {b.id: b for b in s.execute(
        select(ReviewBatch).where(ReviewBatch.id.in_({c.batch_id for c in candidates}))
    ).scalars()}
    need = {}
    for bid, batch in batches.items():
        ids = [c.id for c in candidates if c.batch_id == bid]
        need |= _needed(s, batch, ids)

    pool = [c for c in candidates if need.get(c.id, 0) > 0][:pool_size]
    if not pool:
        return None

    # Weight by how much the item still needs and how uncertain the policy was.
    # Softmax rather than argmax so the propensity is well defined and non-zero
    # for everything in the pool.
    logits = [(need.get(c.id, 1) * 0.5 + c.uncertainty + c.priority * 0.25) / temperature
              for c in pool]
    m = max(logits)
    weights = [math.exp(l - m) for l in logits]
    total = sum(weights)
    probs = [w / total for w in weights]

    rng = random.Random(f"{user.id}:{len(already)}")
    pick = rng.choices(range(len(pool)), weights=probs, k=1)[0]
    item, prob = pool[pick], probs[pick]

    s.add(ReviewAssignment(item_id=item.id, user_id=user.id, sampling_prob=prob))
    if item.status == "pending":
        item.status = "in_review"
    s.flush()
    return item, prob


def presentation_order(item: ReviewItem, user_id: int) -> list[dict]:
    """Shuffle candidates deterministically per (item, reviewer).

    Deterministic so a page refresh does not reorder mid-decision; per-reviewer so
    two reviewers on the same item do not share a position bias. Candidate ids
    travel with the candidates, so the answer recorded is independent of the order
    it was shown in.
    """
    seed = int(hashlib.sha256(f"{item.id}:{user_id}".encode()).hexdigest()[:8], 16)
    shown = list(item.candidates or [])
    random.Random(seed).shuffle(shown)
    return shown


# -------------------------------------------------------------------- recording

def record_annotation(
    s: Session, *, item: ReviewItem, user: User | None, protocol: str,
    choice: str | None = None, ranking: list | None = None,
    rubric_scores: dict | None = None, rationale: str = "",
    confidence: float = 1.0, latency_ms: int = 0,
    annotator_type: str = "human", judge_model: str = "",
) -> Annotation:
    """Append a judgement, superseding this annotator's previous one if any."""
    if user is not None:
        prior = s.execute(
            select(Annotation).where(
                Annotation.item_id == item.id, Annotation.user_id == user.id,
                Annotation.annotator_type == annotator_type,
                Annotation.superseded_by.is_(None))
        ).scalars().all()
    else:
        prior = []

    ann = Annotation(
        item_id=item.id, team_id=item.team_id,
        user_id=user.id if user else None, annotator_type=annotator_type,
        judge_model=judge_model, protocol=protocol, choice=choice,
        ranking=ranking or [], rubric_scores=rubric_scores or {},
        rationale=rationale[:4000], confidence=confidence, latency_ms=latency_ms,
    )
    s.add(ann)
    s.flush()
    for p in prior:
        p.superseded_by = ann.id

    if user is not None:
        assignment = s.execute(
            select(ReviewAssignment).where(ReviewAssignment.item_id == item.id,
                                           ReviewAssignment.user_id == user.id)
        ).scalar_one_or_none()
        if assignment:
            assignment.status = "done"
            assignment.completed_at = utcnow()

    _refresh_item_status(s, item)
    return ann


def _refresh_item_status(s: Session, item: ReviewItem) -> None:
    batch = s.get(ReviewBatch, item.batch_id)
    humans = _live_annotations(s, item.id, "human")
    if len(humans) < (batch.annotations_per_item if batch else 1):
        item.status = "in_review"
        return
    consensus = compute_consensus(humans)
    # Deadlocked items go to a lead rather than being resolved by majority-of-two.
    item.status = "disputed" if consensus["agreement"] < 0.6 and len(humans) > 1 else "complete"


def _live_annotations(s: Session, item_id: int, annotator_type: str | None = None) -> list[Annotation]:
    q = select(Annotation).where(Annotation.item_id == item_id,
                                 Annotation.superseded_by.is_(None))
    if annotator_type:
        q = q.where(Annotation.annotator_type == annotator_type)
    return list(s.execute(q).scalars())


# -------------------------------------------------------------------- consensus

def compute_consensus(annotations: list[Annotation]) -> dict[str, Any]:
    """Majority verdict plus how lopsided it was.

    ``agreement`` is the plurality share — 1.0 when everyone agreed, 0.5 on a
    two-way split. ``margin`` derives from it and becomes the DPO pair weight, so
    a contested item genuinely pushes the policy less far than a unanimous one.
    """
    if not annotations:
        return {"choice": None, "agreement": 0.0, "n": 0, "margin": 0.0}
    votes = Counter(a.choice for a in annotations if a.choice)
    if not votes:
        return {"choice": None, "agreement": 0.0, "n": len(annotations), "margin": 0.0}
    choice, count = votes.most_common(1)[0]
    n = sum(votes.values())
    agreement = count / n
    confidence = sum(a.confidence for a in annotations) / len(annotations)
    return {
        "choice": choice,
        "agreement": agreement,
        "n": n,
        "votes": dict(votes),
        # A unanimous, confident 5-0 gives 1.0; a 3-2 split gives 0.2 before the
        # confidence discount. That spread is the point.
        "margin": round(max(0.0, (2 * agreement - 1)) * confidence, 4),
    }


def resolve_item(s: Session, item: ReviewItem) -> dict[str, Any]:
    """Final verdict: a lead's adjudication if there is one, else the consensus."""
    adj = s.execute(
        select(Adjudication).where(Adjudication.item_id == item.id)
    ).scalar_one_or_none()
    if adj:
        return {"choice": adj.choice, "agreement": 1.0, "margin": 1.0,
                "n": 1, "source": "adjudicated"}

    humans = _live_annotations(s, item.id, "human")
    if humans:
        return compute_consensus(humans) | {"source": "human"}
    ai = _live_annotations(s, item.id, "ai")
    if ai:
        return compute_consensus(ai) | {"source": "ai"}
    return {"choice": None, "agreement": 0.0, "margin": 0.0, "n": 0, "source": "none"}


# ------------------------------------------------------------- pair assembly

def assemble_pairs(s: Session, batch_id: int, *, replace: bool = True) -> dict[str, Any]:
    """Turn resolved review items into preference pairs the optimisers can eat."""
    batch = s.get(ReviewBatch, batch_id)
    if batch is None:
        raise ValueError(f"no review batch {batch_id}")

    if replace:
        for old in s.execute(
            select(PreferencePair).where(PreferencePair.batch_id == batch_id)
        ).scalars():
            s.delete(old)
        s.flush()

    items = list(s.execute(
        select(ReviewItem).where(ReviewItem.batch_id == batch_id)
    ).scalars())

    made, ties, unresolved = 0, 0, 0
    for item in items:
        verdict = resolve_item(s, item)
        if not verdict["choice"]:
            unresolved += 1
            continue
        if verdict["choice"] in TIE_CHOICES:
            ties += 1
            continue

        by_id = {c["id"]: c for c in (item.candidates or [])}
        winner = by_id.get(verdict["choice"])
        if winner is None:
            unresolved += 1
            continue

        for cid, cand in by_id.items():
            if cid == verdict["choice"]:
                continue
            if cand["text"].strip() == winner["text"].strip():
                continue
            s.add(PreferencePair(
                team_id=item.team_id, batch_id=batch_id, item_id=item.id,
                prompt=item.prompt, chosen=winner["text"], rejected=cand["text"],
                # An adjudication is a (senior) human's call, so it counts as human.
                source="human" if verdict["source"] in ("human", "adjudicated") else "ai",
                margin=verdict["margin"] or 1.0,
                agreement=verdict["agreement"],
                # Inverse-propensity weight: items the queue oversampled must not
                # also count extra in the loss, or the bias the sampler introduced
                # deliberately gets baked into the model permanently.
                weight=round(_ipw(s, item.id), 4),
            ))
            made += 1

    s.flush()
    return {"pairs": made, "ties_dropped": ties, "unresolved": unresolved,
            "items": len(items)}


# Propensity a uniformly-sampled item would have had, given the default pool.
# Weights are expressed relative to it so that an unbiased queue yields w = 1.0
# and the numbers stay interpretable.
UNIFORM_PROPENSITY = 1 / 40


def _ipw(s: Session, item_id: int, lo: float = 0.2, hi: float = 5.0) -> float:
    """Inverse-propensity weight, normalised and clipped.

    w = p_uniform / p_actual, so an item the queue favoured 5x carries 1/5 the
    weight and an item it almost skipped carries more. The clip is not cosmetic:
    an item sampled with p=0.001 would otherwise arrive with forty times its
    neighbours' weight and decide the update on its own.
    """
    probs = list(s.execute(
        select(ReviewAssignment.sampling_prob).where(ReviewAssignment.item_id == item_id)
    ).scalars())
    if not probs:
        return 1.0
    p = sum(probs) / len(probs)
    if p <= 0:
        return hi
    return max(lo, min(hi, UNIFORM_PROPENSITY / p))


# ---------------------------------------------------------------- diagnostics

def agreement_report(s: Session, team_id: int, batch_id: int | None = None) -> dict[str, Any]:
    """Inter-annotator agreement, and how well the judge tracks the humans.

    The human-vs-AI number is the one that decides whether RLAIF is trustworthy on
    this task. Reported as raw agreement *and* as Cohen's kappa, because raw
    agreement on a two-way choice starts at 50% for a pair of coin flips and looks
    respectable when it means nothing.
    """
    q = select(ReviewItem).where(ReviewItem.team_id == team_id)
    if batch_id:
        q = q.where(ReviewItem.batch_id == batch_id)
    items = list(s.execute(q).scalars())

    pairwise_hits = pairwise_total = 0
    ai_hits = ai_total = 0
    human_dist: Counter = Counter()
    ai_dist: Counter = Counter()
    disputed = 0

    for item in items:
        humans = [a for a in _live_annotations(s, item.id, "human") if a.choice]
        ai = [a for a in _live_annotations(s, item.id, "ai") if a.choice]

        for i in range(len(humans)):
            for j in range(i + 1, len(humans)):
                pairwise_total += 1
                pairwise_hits += int(humans[i].choice == humans[j].choice)

        if humans and ai:
            consensus = compute_consensus(humans)["choice"]
            ai_total += 1
            ai_hits += int(consensus == ai[0].choice)
            human_dist[consensus] += 1
            ai_dist[ai[0].choice] += 1

        if item.status == "disputed":
            disputed += 1

    return {
        "items": len(items),
        "disputed": disputed,
        "inter_annotator_agreement": round(pairwise_hits / pairwise_total, 4) if pairwise_total else None,
        "comparisons": pairwise_total,
        "human_vs_ai_agreement": round(ai_hits / ai_total, 4) if ai_total else None,
        "human_vs_ai_kappa": _kappa(human_dist, ai_dist, ai_hits, ai_total),
        "human_vs_ai_n": ai_total,
        "annotator_throughput": _throughput(s, team_id, batch_id),
    }


def _kappa(dist_a: Counter, dist_b: Counter, hits: int, total: int) -> float | None:
    """Cohen's kappa: agreement above what the two label distributions would
    produce by chance alone."""
    if not total:
        return None
    po = hits / total
    labels = set(dist_a) | set(dist_b)
    pe = sum((dist_a.get(l, 0) / total) * (dist_b.get(l, 0) / total) for l in labels)
    if pe >= 1.0:
        return None
    return round((po - pe) / (1 - pe), 4)


def _throughput(s: Session, team_id: int, batch_id: int | None) -> list[dict]:
    q = (select(Annotation.user_id, func.count(), func.avg(Annotation.latency_ms))
         .where(Annotation.team_id == team_id, Annotation.annotator_type == "human",
                Annotation.superseded_by.is_(None))
         .group_by(Annotation.user_id))
    rows = s.execute(q).all()
    users = {u.id: u.name for u in s.execute(select(User).where(User.team_id == team_id)).scalars()}
    return [
        {"user_id": uid, "name": users.get(uid, "?"), "annotations": n,
         "median_seconds": round((avg or 0) / 1000, 1)}
        for uid, n, avg in sorted(rows, key=lambda r: -r[1])
    ]
