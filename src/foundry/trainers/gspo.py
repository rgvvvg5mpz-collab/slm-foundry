"""GSPO entry point. The algorithm lives in :mod:`.group_rl`, shared with GRPO."""
from __future__ import annotations

from .base import TrainContext, TrainRequest, TrainResult
from .group_rl import run_group_rl


def run(ctx: TrainContext, req: TrainRequest, backend: str) -> TrainResult:
    return run_group_rl(ctx, req, backend, mode="gspo")
