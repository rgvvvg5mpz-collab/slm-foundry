"""One place that decides which model a trainer gets.

Every trainer calls :func:`build_policy`; none of them contain an ``if backend ==``.
That keeps the objective modules readable and means the tiny-model fallback can
never be half-applied — a trainer cannot accidentally load a real base model for
its policy and a tiny one for its reference.
"""
from __future__ import annotations

from typing import Any

from .base import TrainRequest
from .policy import Policy, load_policy
from .tiny import load_tiny_policy


def build_policy(req: TrainRequest, backend: str, *, adapter_dir: str | None = None,
                 inference_only: bool = False) -> Policy:
    adapter = adapter_dir if adapter_dir is not None else req.policy_adapter_dir
    if backend == "tiny":
        corpus = (req.train_rows or []) + (req.val_rows or [])
        return load_tiny_policy(corpus, req.lora, adapter, seed=req.seed)
    return load_policy(req.base_model, req.lora, adapter, inference_only=inference_only)


def prompts_and_targets(policy: Policy, rows: list[dict], target_field: str) -> tuple[list[str], list[str]]:
    from .policy import render
    prompts = [render(policy.tokenizer, r.get("messages", [])) for r in rows]
    targets = [r.get(target_field) or "" for r in rows]
    return prompts, targets


def prompt_texts(policy: Policy, rows: list[dict]) -> list[str]:
    from .policy import render
    return [render(policy.tokenizer, r.get("messages", [])) for r in rows]


def optimizer_for(params, lr: float, weight_decay: float = 0.0):
    import torch
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))


def scheduler_for(optimizer, total_steps: int, warmup_ratio: float):
    import torch
    warmup = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def describe(policy: Policy, backend: str) -> dict[str, Any]:
    return {
        "backend": backend,
        "base_model": policy.base_model,
        "device": str(policy.device),
        "lora": policy.lora_info,
    }
