"""Training methods, one module each, behind a single dispatch table.

Every trainer has the same signature — ``run(ctx, req) -> TrainResult`` — and the
worker knows nothing about which one it is executing. Adding a method means
adding a module and one catalogue entry in ``configs/foundry.json``; it does not
mean touching the queue, the API, or the UI.
"""
from __future__ import annotations

from .base import Cancelled, TrainContext, TrainRequest, TrainResult


def get_trainer(method: str):
    from . import dpo, grpo, gspo, ppo, reward, rlaif, sft
    table = {
        "sft": sft.run,
        "reward": reward.run,
        "dpo": dpo.run,
        "ppo": ppo.run,
        "grpo": grpo.run,
        "gspo": gspo.run,
        "rlaif": rlaif.run,
    }
    if method not in table:
        raise KeyError(f"no trainer for method {method!r}")
    return table[method]


__all__ = ["Cancelled", "TrainContext", "TrainRequest", "TrainResult", "get_trainer"]
