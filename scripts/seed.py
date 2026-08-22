"""Seed a demo tenant: two teams, five users, four datasets.

Run once against an empty database::

    PYTHONPATH=src python scripts/seed.py

The datasets are synthetic but structurally realistic — a policy Q&A assistant,
which is the shape most internal SLM projects actually take. They exist so that
every screen in the product has something in it on first login, and so the smoke
test has fixtures that exercise all four dataset kinds.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foundry import auth as authlib                                  # noqa: E402
from foundry import datasets as dslib                                # noqa: E402
from foundry.db import get_session, init_db                          # noqa: E402
from foundry.models import Dataset, Team, User                       # noqa: E402
from foundry.paths import upload_dir                                 # noqa: E402

TOPICS = [
    ("expense reimbursement", "Receipts are required for anything above $25, and claims must be "
     "filed within 60 days of the expense."),
    ("parental leave", "Sixteen weeks at full pay for all new parents, available any time within "
     "the first year."),
    ("remote work", "Fully remote is available with manager approval; hybrid staff are expected "
     "onsite two days a week."),
    ("data retention", "Customer records are kept for seven years, then deleted automatically "
     "unless a legal hold applies."),
    ("security incidents", "Report anything suspicious to security@ within one hour of noticing "
     "it — an incorrect report costs nothing, a late one can cost a great deal."),
    ("procurement", "Anything above $5,000 needs two quotes and a director's sign-off before the "
     "purchase order is raised."),
    ("contractor onboarding", "Contractors need a signed SOW and a background check before system "
     "access is granted."),
    ("travel booking", "Book through the corporate portal; out-of-policy fares need written "
     "approval in advance."),
    ("device replacement", "Laptops are refreshed every three years, or sooner if repair costs "
     "exceed half the replacement price."),
    ("training budget", "Each employee has $2,000 per year for external training, which does not "
     "roll over."),
]

QUESTION_FORMS = [
    "What's the policy on {t}?",
    "Can you explain how {t} works here?",
    "I have a question about {t} — what do I need to know?",
    "Quick one: {t}. What are the rules?",
    "A new joiner asked me about {t}. What should I tell them?",
]

WEAK_FORMS = [
    "I think there's a policy about {t} but I'm not sure of the details. Check the handbook.",
    "Policy policy policy. {t}. Ask someone else.",
    "Yes.",
    "That depends on a lot of things and it would be hard to say without more context, though "
    "generally speaking there may well be a relevant policy covering {t} in some form.",
]


def build_corpora(seed: int = 7) -> dict[str, bytes]:
    rng = random.Random(seed)

    # Every prompt carries a distinguishing detail. Without one, ten topics × five
    # phrasings is fifty unique rows however many are generated, and the ingest's
    # deduplication — correctly — throws the rest away.
    ROLES = ["a new joiner", "a contractor", "my manager", "someone in finance",
             "a team lead", "an intern", "a remote colleague"]

    sft, prompts, bench, prefs = [], [], [], []
    for i in range(140):
        topic, answer = TOPICS[i % len(TOPICS)]
        q = (rng.choice(QUESTION_FORMS).format(t=topic)
             + f" Asking on behalf of {ROLES[i % len(ROLES)]}.")
        sft.append({"messages": [
            {"role": "system", "content": "You are the internal policy assistant. Answer from "
                                          "policy, be specific, and say when you are unsure."},
            {"role": "user", "content": q}], "completion": answer})

    for i in range(60):
        topic, _ = TOPICS[i % len(TOPICS)]
        prompts.append({"prompt": rng.choice(QUESTION_FORMS).format(t=topic)
                        + f" Context: {ROLES[i % len(ROLES)]}.",
                        "meta": {"topic": topic}})

    for i in range(40):
        topic, answer = TOPICS[i % len(TOPICS)]
        row = {"prompt": rng.choice(QUESTION_FORMS).format(t=topic)
               + f" ({ROLES[i % len(ROLES)]})", "reference": answer}
        if i % 4 == 0:
            distractor = TOPICS[(i + 3) % len(TOPICS)][1]
            row |= {"choices": [answer, distractor], "answer": answer}
        bench.append(row)

    for i in range(70):
        topic, answer = TOPICS[i % len(TOPICS)]
        prefs.append({
            "prompt": rng.choice(QUESTION_FORMS).format(t=topic)
                      + f" — for {ROLES[i % len(ROLES)]}.",
            "chosen": answer,
            "rejected": rng.choice(WEAK_FORMS).format(t=topic),
            "margin": round(rng.uniform(0.6, 1.0), 2),
        })

    enc = lambda rows: ("\n".join(json.dumps(r) for r in rows)).encode()
    return {"sft": enc(sft), "prompts": enc(prompts),
            "benchmark": enc(bench), "preference": enc(prefs)}


USERS = [
    ("admin@foundry.local", "Ada Founder", "admin"),
    ("lead@foundry.local", "Lena Lead", "lead"),
    ("ml@foundry.local", "Marco Engineer", "member"),
    ("reviewer@foundry.local", "Ren Reviewer", "reviewer"),
    ("reviewer2@foundry.local", "Robin Second", "reviewer"),
]
PASSWORD = "foundry-demo"


def seed(reset: bool = False) -> None:
    init_db(drop=reset)
    with get_session() as s:
        if s.query(Team).count():
            print("database already seeded — pass --reset to start over")
            return

        core = Team(slug="policy-ai", name="Policy AI", concurrency=2)
        other = Team(slug="support-ai", name="Support AI", concurrency=1)
        s.add_all([core, other])
        s.flush()

        for email, name, role in USERS:
            s.add(User(email=email, name=name, role=role, team_id=core.id,
                       password_hash=authlib.hash_password(PASSWORD)))
        s.add(User(email="other@foundry.local", name="Sam Neighbour", role="member",
                   team_id=other.id, password_hash=authlib.hash_password(PASSWORD)))
        s.flush()

        for kind, raw in build_corpora().items():
            ds = Dataset(team_id=core.id, name=f"policy-{kind}", version=1, kind=kind,
                         description=f"Synthetic {kind} corpus for the policy assistant.",
                         status="validating")
            s.add(ds)
            s.flush()
            dest = upload_dir(core.slug, ds.id) / "data.jsonl"
            report = dslib.ingest(raw, kind, dest)
            ds.path = str(dest)
            ds.status = report["status"]
            ds.num_rows = report["num_rows"]
            ds.num_bad_rows = report["num_bad_rows"]
            ds.bytes = report["bytes"]
            ds.sha256 = report["sha256"]
            ds.stats = report["stats"]
            ds.errors = report["errors"]
            print(f"  {kind:11s} {report['num_rows']:4d} rows → {dest}")

    print("\nseeded. sign in as any of:")
    for email, name, role in USERS:
        print(f"  {email:28s} {PASSWORD}   ({role})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="drop every table first")
    seed(ap.parse_args().reset)
