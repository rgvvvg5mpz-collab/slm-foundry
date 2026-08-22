"""The ``tiny`` backend: a randomly-initialised micro-model fitted to your data.

This is *not* a mock. Every objective in this package — the DPO log-ratio, the
PPO clipped surrogate with GAE, GRPO's group-normalised advantage, GSPO's
sequence-level ratio — runs unchanged against a real ``GPT2LMHeadModel``. What
changes is its size: two layers, 64 hidden units, and a word-level vocabulary
built from the uploaded dataset, which trains on a laptop CPU in seconds and
needs no download.

Why it exists, in order of importance:

1. **The pipeline is verifiable.** A four-minute end-to-end test that exercises
   all seven methods for real catches an off-by-one in a log-prob mask. A test
   that stubs the model out catches nothing.
2. **The product is inspectable without a GPU.** A team evaluating SLM Foundry
   can upload data, run every pipeline, review outputs and promote a model
   before anyone provisions hardware.
3. **New pipelines get a cheap smoke target.**

What it is not: a source of meaningful quality numbers. A 117k-parameter model
initialised from noise produces loss curves that are real and generations that
are gibberish. Every run records ``backend`` and the UI labels tiny-backend runs
explicitly, because a metrics chart from this must never be read as a result.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers as tk_trainers
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

from . import lora as loralib
from .policy import Policy, pick_device

VOCAB_SIZE = 4096
N_POSITIONS = 512


def _text_corpus(rows: list[dict]) -> Iterable[str]:
    """Every string in the dataset, whatever its shape."""
    for row in rows:
        for m in row.get("messages", []):
            yield m.get("content", "")
        for field in ("completion", "chosen", "rejected", "reference", "rubric"):
            v = row.get(field)
            if isinstance(v, str):
                yield v
        for c in row.get("choices") or []:
            yield str(c)


def build_tokenizer(rows: list[dict]) -> PreTrainedTokenizerFast:
    corpus = [t for t in _text_corpus(rows) if t] or ["placeholder text"]
    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(
        corpus,
        tk_trainers.WordLevelTrainer(
            vocab_size=VOCAB_SIZE,
            special_tokens=["<pad>", "<unk>", "<eos>", "System:", "User:", "Assistant:"],
        ),
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>",
        eos_token="<eos>", bos_token="<eos>",
    )
    fast.padding_side = "right"
    return fast


def load_tiny_policy(rows: list[dict], lora_cfg: dict[str, Any] | None = None,
                     adapter_dir: str | None = None, *, seed: int = 0) -> Policy:
    torch.manual_seed(seed)
    tokenizer = build_tokenizer(rows)
    config = AutoConfig.for_model(
        "gpt2",
        vocab_size=len(tokenizer),
        n_embd=64, n_layer=2, n_head=2, n_positions=N_POSITIONS,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = AutoModelForCausalLM.from_config(config)
    # CPU regardless of what the box has: moving a 117k-parameter model to a GPU
    # costs more in transfer than it saves in matmul, and MPS lacks kernels for
    # some of the ops the RL path uses.
    device = torch.device("cpu")
    model.to(device)

    cfg = lora_cfg or {}
    info = loralib.inject(
        model,
        r=int(cfg.get("r", 8)),
        alpha=float(cfg.get("alpha", 16)),
        dropout=float(cfg.get("dropout", 0.0)),
        targets=["c_attn", "c_proj", "c_fc"],
    )
    info["tiny"] = True
    if adapter_dir:
        try:
            loralib.load(model, adapter_dir)
            info["loaded_from"] = str(adapter_dir)
        except Exception as e:
            # A tiny model's vocabulary is derived from the dataset, so an adapter
            # trained on a different dataset genuinely does not fit. Say so and
            # continue from scratch rather than failing the run.
            info["adapter_load_skipped"] = str(e)[:200]

    return Policy(model=model, tokenizer=tokenizer, device=device,
                  lora_info=info, base_model="tiny-gpt2 (random init)")
