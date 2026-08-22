"""Torch-side model handling: load, tokenize, score, sample.

Everything the preference and RL trainers need from a language model lives here,
so the objective modules contain objective maths and nothing else. Three pieces
carry most of the weight:

``build_batch``  turns (prompt, completion) pairs into padded tensors with a
label mask that covers *only* the completion. Prompt tokens contributing to the
loss is the single most common quiet bug in a fine-tuning stack — the model
learns to generate your prompts and nobody notices until the eval looks strange.

``token_logprobs``  returns per-token log-probabilities aligned to that mask.
DPO sums them, PPO/GRPO use them per token, GSPO length-normalises them. One
implementation, so the three cannot disagree about off-by-one alignment.

``generate``  samples completions and returns them already tokenized alongside
their masks, because re-tokenizing decoded text does not round-trip and a
mismatch between what was sampled and what is scored silently corrupts the
importance ratio.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import lora as loralib


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32          # mps/cpu: fp32 is both faster and less surprising


@dataclass
class Policy:
    model: Any
    tokenizer: Any
    device: torch.device
    lora_info: dict[str, Any]
    base_model: str

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save(self, out_dir: Path, meta: dict[str, Any] | None = None) -> Path:
        return loralib.save(self.model, out_dir, {
            "base_model": self.base_model, "lora": self.lora_info, **(meta or {})})


def load_policy(
    base_model: str,
    lora_cfg: dict[str, Any] | None = None,
    adapter_dir: str | None = None,
    *,
    inference_only: bool = False,
) -> Policy:
    device = pick_device()
    dtype = pick_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        # Padding with EOS is fine because the attention mask excludes pads; what
        # is not fine is leaving pad_token unset and getting a silent crash deep
        # in the collator on the first ragged batch.
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
    model.to(device)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    cfg = lora_cfg or {}
    info = loralib.inject(
        model,
        r=int(cfg.get("r", 16)),
        alpha=float(cfg.get("alpha", 32)),
        dropout=0.0 if inference_only else float(cfg.get("dropout", 0.05)),
        targets=cfg.get("targets") or None,
    )
    if adapter_dir:
        info["loaded_from"] = str(adapter_dir)
        loralib.load(model, adapter_dir)
    if inference_only:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

    return Policy(model=model, tokenizer=tokenizer, device=device,
                  lora_info=info, base_model=base_model)


# ----------------------------------------------------------------- prompt text

def render(tokenizer, messages: list[dict]) -> str:
    """Apply the model's chat template when it has one, else a plain transcript."""
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    from ..datasets import render_prompt
    return render_prompt(messages)


# -------------------------------------------------------------------- batching

@dataclass
class Batch:
    input_ids: torch.Tensor        # (B, T)
    attention_mask: torch.Tensor   # (B, T)
    completion_mask: torch.Tensor  # (B, T-1) aligned to next-token predictions
    prompt_lens: list[int]


def build_batch(
    tokenizer,
    device: torch.device,
    prompts: list[str],
    completions: list[str],
    max_seq_len: int,
    *,
    completion_only: bool = True,
) -> Batch:
    ids_list, masks = [], []
    prompt_lens: list[int] = []
    eos = tokenizer.eos_token_id

    for prompt, completion in zip(prompts, completions):
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        if eos is not None:
            c_ids = c_ids + [eos]
        # Truncate the prompt from the left, not the completion: the target is the
        # part we are training on, and clipping its tail teaches the model to stop
        # mid-sentence.
        budget = max_seq_len - len(c_ids)
        if budget < 1:
            c_ids = c_ids[: max_seq_len - 1]
            budget = 1
        p_ids = p_ids[-budget:]
        ids = p_ids + c_ids
        ids_list.append(ids)
        prompt_lens.append(len(p_ids))
        masks.append([0] * len(p_ids) + [1] * len(c_ids) if completion_only else [1] * len(ids))

    width = max(len(i) for i in ids_list)
    pad = tokenizer.pad_token_id
    input_ids = torch.full((len(ids_list), width), pad, dtype=torch.long)
    attention = torch.zeros((len(ids_list), width), dtype=torch.long)
    label_mask = torch.zeros((len(ids_list), width), dtype=torch.float)
    for i, (ids, m) in enumerate(zip(ids_list, masks)):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        attention[i, : len(ids)] = 1
        label_mask[i, : len(m)] = torch.tensor(m, dtype=torch.float)

    return Batch(
        input_ids=input_ids.to(device),
        attention_mask=attention.to(device),
        # Shifted: position t of this mask selects the prediction of token t+1.
        completion_mask=label_mask[:, 1:].to(device),
        prompt_lens=prompt_lens,
    )


# ------------------------------------------------------------------- log-probs

def token_logprobs(model, batch: Batch, *, requires_grad: bool = True) -> torch.Tensor:
    """(B, T-1) log p(token_{t+1} | ..t). Zero wherever the completion mask is zero."""
    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx:
        logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
        logits = logits[:, :-1, :].float()
        targets = batch.input_ids[:, 1:]
        logp = torch.log_softmax(logits, dim=-1)
        picked = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return picked * batch.completion_mask


def sequence_logprob(model, batch: Batch, *, requires_grad: bool = True,
                     average: bool = False) -> torch.Tensor:
    """(B,) summed — or length-averaged — completion log-probability.

    ``average=True`` is what separates GSPO from GRPO and what IPO uses: it makes
    the quantity comparable across completions of very different lengths.
    """
    tl = token_logprobs(model, batch, requires_grad=requires_grad)
    total = tl.sum(dim=-1)
    if not average:
        return total
    lengths = batch.completion_mask.sum(dim=-1).clamp(min=1)
    return total / lengths


def reference_token_logprobs(model, batch: Batch) -> torch.Tensor:
    """Log-probs under the frozen base model — adapters off, no gradient."""
    with loralib.adapters_disabled(model), torch.no_grad():
        return token_logprobs(model, batch, requires_grad=False)


def entropy_from_logits(model, batch: Batch) -> torch.Tensor:
    with torch.no_grad():
        logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
        logits = logits[:, :-1, :].float()
        p = torch.softmax(logits, dim=-1)
        ent = -(p * torch.log_softmax(logits, dim=-1)).sum(-1)
    denom = batch.completion_mask.sum().clamp(min=1)
    return (ent * batch.completion_mask).sum() / denom


# ------------------------------------------------------------------ generation

@torch.no_grad()
def generate(
    policy: Policy,
    prompts: list[str],
    *,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0,
    num_return_sequences: int = 1,
) -> list[list[str]]:
    """Sample completions. Returns one list of strings per prompt."""
    tok = policy.tokenizer
    was_training = policy.model.training
    policy.model.eval()
    # Left padding so every sequence's last real token sits at the same index —
    # right padding here makes the model continue from pad tokens.
    old_side, tok.padding_side = tok.padding_side, "left"
    try:
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024, add_special_tokens=False).to(policy.device)
        out = policy.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=tok.pad_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(gen, skip_special_tokens=True)
    finally:
        tok.padding_side = old_side
        if was_training:
            policy.model.train()

    return [texts[i * num_return_sequences:(i + 1) * num_return_sequences]
            for i in range(len(prompts))]


# --------------------------------------------------------------- value head (PPO)

class ValueHead(torch.nn.Module):
    """Scalar critic over the policy's hidden states.

    Kept as a separate module rather than a second model: PPO needs V(s_t) at every
    token of the same forward pass the policy log-probs come from, and running a
    whole second network to produce one number per position is wasted compute.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.1)
        self.proj = torch.nn.Linear(hidden_size, 1)
        torch.nn.init.normal_(self.proj.weight, std=1 / (hidden_size + 1) ** 0.5)
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(self.dropout(hidden.float())).squeeze(-1)


def policy_forward_with_values(model, value_head: ValueHead, batch: Batch):
    """One pass yielding both per-token log-probs and per-token values."""
    out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                output_hidden_states=True)
    logits = out.logits[:, :-1, :].float()
    targets = batch.input_ids[:, 1:]
    logp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    values = value_head(out.hidden_states[-1][:, :-1, :])
    return logp * batch.completion_mask, values * batch.completion_mask


def masked_whiten(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    n = mask.sum().clamp(min=1)
    mean = (x * mask).sum() / n
    var = (((x - mean) ** 2) * mask).sum() / n
    return ((x - mean) * torch.rsqrt(var + eps)) * mask


def kl_k3(logp: torch.Tensor, ref_logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Schulman's low-variance, always-positive KL estimator.

    exp(d) - d - 1 with d = log π_ref - log π. The naive estimator (-d) is
    unbiased but routinely goes negative on a batch, which makes a KL *penalty*
    occasionally pay the policy to move away from the reference.
    """
    d = (ref_logp - logp) * mask
    return (torch.exp(d) - d - 1.0) * mask


# ------------------------------------------------------------- reward modelling

class RewardModel(torch.nn.Module):
    """Backbone + scalar head, scored at the last real token of the sequence.

    "Last real token" rather than mean-pooling is deliberate: a reward model that
    averages over positions can be gamed by padding a good answer with filler,
    because each additional bland token drags the mean toward whatever the model
    considers neutral. Reading the final position forces it to commit to a verdict
    on the completed response.
    """

    def __init__(self, policy: Policy):
        super().__init__()
        self.policy = policy
        hidden = getattr(policy.model.config, "hidden_size", None) or policy.model.config.n_embd
        self.head = torch.nn.Linear(hidden, 1)
        torch.nn.init.normal_(self.head.weight, std=1 / (hidden + 1) ** 0.5)
        torch.nn.init.zeros_(self.head.bias)
        self.head.to(policy.device).to(torch.float32)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.policy.model(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=True)
        hidden = out.hidden_states[-1].float()
        last = attention_mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
        return self.head(pooled).squeeze(-1)

    def score(self, prompts: list[str], completions: list[str], max_seq_len: int = 1024,
              batch_size: int = 8) -> list[float]:
        self.eval()
        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch = build_batch(self.policy.tokenizer, self.policy.device,
                                    prompts[i:i + batch_size], completions[i:i + batch_size],
                                    max_seq_len)
                scores.extend(self(batch.input_ids, batch.attention_mask).cpu().tolist())
        return scores

    def save(self, out_dir: Path, meta: dict[str, Any] | None = None) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.policy.save(out_dir, {"artifact_kind": "reward", **(meta or {})})
        torch.save(self.head.state_dict(), out_dir / "reward_head.pt")
        return out_dir

    def load_head(self, adapter_dir: str | Path) -> None:
        self.head.load_state_dict(torch.load(Path(adapter_dir) / "reward_head.pt",
                                             map_location=self.policy.device))
