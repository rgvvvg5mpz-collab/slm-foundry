"""SLM Foundry — bring your data, get a fine-tuned small language model.

Four moving parts, one loop:

    datasets  →  SFT (LoRA)  →  review (human or AI)  →  RLHF/RLAIF  →  registry
                     ↑                                                     │
                     └─────────────────── promote / branch ────────────────┘

Everything long-running goes through :mod:`foundry.queue` as a *job*, and jobs are
executed by :mod:`foundry.worker` processes. Nothing heavy ever runs inside a
request handler — the API stays responsive when six teams all launch a run at
09:00 on Monday, and a worker crash costs one job rather than the web tier.
"""

__version__ = "1.0.0"
