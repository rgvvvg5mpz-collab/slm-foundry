"""Supervised fine-tuning with LoRA.

The whole method is one idea — maximise the likelihood of the demonstrated
response given the prompt — so the interesting decisions are all in the details:

* **Loss on the completion only** (default on). With the prompt included, the
  model spends most of its gradient learning to generate your instructions, which
  is capacity spent on something you will never ask it to do.
* **Validation split by prompt hash**, so growing the dataset does not reshuffle
  the split and leak yesterday's validation set into today's training data.
* **Token accuracy alongside loss.** Cross-entropy answers "how surprised was the
  model"; token accuracy answers "would it have written this", and the two come
  apart in ways worth seeing on the same chart.
"""
from __future__ import annotations

import math
from typing import Any

import torch

from .base import TrainContext, TrainRequest, TrainResult
from .factory import build_policy, describe, optimizer_for, prompts_and_targets, scheduler_for
from .policy import build_batch, generate, token_logprobs


def _evaluate(policy, rows, params, max_batches: int = 8) -> dict[str, float]:
    if not rows:
        return {}
    prompts, targets = prompts_and_targets(policy, rows, "completion")
    bs = int(params["batch_size"])
    policy.eval()
    total_nll, total_tokens, total_correct = 0.0, 0, 0
    with torch.no_grad():
        for i in range(0, min(len(rows), bs * max_batches), bs):
            batch = build_batch(policy.tokenizer, policy.device, prompts[i:i + bs],
                                targets[i:i + bs], int(params["max_seq_len"]),
                                completion_only=bool(params["completion_only_loss"]))
            logits = policy.model(input_ids=batch.input_ids,
                                  attention_mask=batch.attention_mask).logits[:, :-1, :].float()
            tgt = batch.input_ids[:, 1:]
            lp = torch.log_softmax(logits, dim=-1)
            picked = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            mask = batch.completion_mask
            total_nll += float(-(picked * mask).sum())
            total_tokens += int(mask.sum())
            total_correct += int(((logits.argmax(-1) == tgt).float() * mask).sum())
    policy.train()
    if total_tokens == 0:
        return {}
    nll = total_nll / total_tokens
    return {
        "val_loss": nll,
        "val_ppl": float(math.exp(min(nll, 20))),
        "val_token_acc": total_correct / total_tokens,
    }


def run(ctx: TrainContext, req: TrainRequest, backend: str) -> TrainResult:
    params = req.params
    ctx.log(f"SFT: {len(req.train_rows)} train / {len(req.val_rows)} val rows")

    policy = build_policy(req, backend)
    ctx.log("policy ready", **describe(policy, backend))
    policy.train()

    prompts, targets = prompts_and_targets(policy, req.train_rows, "completion")
    bs = int(params["batch_size"])
    accum = int(params["grad_accum"])
    steps_per_epoch = max(1, math.ceil(len(prompts) / bs))
    total_steps = max(1, int(steps_per_epoch * float(params["epochs"])))

    opt = optimizer_for([p for p in policy.model.parameters() if p.requires_grad],
                        float(params["learning_rate"]), float(params["weight_decay"]))
    sched = scheduler_for(opt, total_steps, float(params["warmup_ratio"]))

    ctx.log(f"{total_steps} optimizer steps "
            f"({steps_per_epoch}/epoch × {params['epochs']} epochs, batch {bs}×{accum})")

    order = list(range(len(prompts)))
    g = torch.Generator().manual_seed(req.seed)
    best_val = float("inf")
    eval_every = max(1, total_steps // 10)

    for step in range(1, total_steps + 1):
        if (step - 1) % steps_per_epoch == 0:
            order = torch.randperm(len(prompts), generator=g).tolist()

        opt.zero_grad(set_to_none=True)
        step_loss, step_tokens = 0.0, 0
        for micro in range(accum):
            start = ((step - 1) * accum + micro) * bs % max(1, len(order))
            idx = [order[(start + k) % len(order)] for k in range(min(bs, len(order)))]
            batch = build_batch(
                policy.tokenizer, policy.device,
                [prompts[i] for i in idx], [targets[i] for i in idx],
                int(params["max_seq_len"]),
                completion_only=bool(params["completion_only_loss"]),
            )
            tl = token_logprobs(policy.model, batch)
            ntokens = batch.completion_mask.sum().clamp(min=1)
            # Token-mean, not sequence-mean: with sequence-mean a 400-token answer
            # and a 4-token answer contribute equally per token, which quietly
            # up-weights short examples by two orders of magnitude.
            loss = -(tl.sum() / ntokens) / accum
            loss.backward()
            step_loss += loss.detach().item() * accum
            step_tokens += int(ntokens)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in policy.model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()

        metrics: dict[str, Any] = {
            "loss": step_loss / accum,
            "ppl": math.exp(min(step_loss / accum, 20)),
            "lr": sched.get_last_lr()[0],
            "grad_norm": float(grad_norm),
            "tokens": step_tokens,
        }
        if step % eval_every == 0 or step == total_steps:
            ev = _evaluate(policy, req.val_rows, params)
            metrics.update(ev)
            if ev.get("val_loss", float("inf")) < best_val:
                best_val = ev["val_loss"]
        ctx.metric(step, total_steps, **metrics)

    adapter_dir = ctx.out_dir / "adapter"
    policy.save(adapter_dir, {"method": "sft", "run_id": req.run_id, "backend": backend})
    ctx.log(f"adapter saved to {adapter_dir}")

    samples = _sample_outputs(policy, req, params, ctx)
    summary = ctx.summarise()
    summary.update({"backend": backend, "train_rows": len(req.train_rows),
                    "val_rows": len(req.val_rows),
                    "trainable_params": policy.lora_info["trainable_params"]})
    if best_val < float("inf"):
        summary["best_val_loss"] = round(best_val, 6)

    return TrainResult(metrics=summary, artifact_dir=str(adapter_dir),
                       artifact_kind="adapter", samples=samples, backend=backend)


def _sample_outputs(policy, req, params, ctx, n: int = 5) -> list[dict]:
    """Generate on a few validation prompts so the run page shows behaviour, not
    just a loss curve. A model whose loss fell but whose outputs are worse is a
    thing that happens, and it is only visible here."""
    rows = (req.val_rows or req.train_rows)[:n]
    if not rows:
        return []
    prompts, targets = prompts_and_targets(policy, rows, "completion")
    try:
        outs = generate(policy, prompts, max_new_tokens=min(96, int(params["max_seq_len"]) // 4),
                        temperature=0.7)
    except Exception as e:                                  # never fail a run on sampling
        ctx.log(f"sample generation skipped: {e}", level="warn")
        return []
    samples = [{"prompt": p, "generated": o[0], "reference": t}
               for p, o, t in zip(prompts, outs, targets)]
    ctx.write_jsonl("samples.jsonl", samples)
    return samples
