"""LoRA: low-rank adapters injected into a frozen base model.

Implemented directly rather than pulled from `peft` for two reasons that matter
here. First, the preference trainers need a *reference* policy on every step, and
the cheapest correct reference is this model with the adapters switched off —
which needs a toggle this module owns (see :func:`adapters_disabled`). Cloning a
second copy of the base model instead would roughly double memory for no gain.
Second, adapter-only checkpoints are what the registry stores, so the save format
is part of the product's contract and worth keeping visible.

The maths is the original formulation: a frozen ``W`` gets ``W + (alpha/r)·B·A``
with ``A`` Kaiming-initialised and ``B`` zeroed, so training starts *exactly* at
the base model's behaviour. Zero-init on B is not a detail — initialise it
randomly and step zero is already a damaged model.
"""
from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn

# Common attention/MLP projection names across the small-model families people
# actually fine-tune. Order does not matter; matching is by name suffix.
DEFAULT_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",          # Llama / Qwen / Mistral / SmolLM
    "gate_proj", "up_proj", "down_proj",
    "c_attn", "c_proj", "c_fc",                       # GPT-2 family (Conv1D)
    "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h",   # Falcon / GPT-NeoX
]


def _shape_of(base: nn.Module) -> tuple[int, int]:
    """(in_features, out_features) for Linear and for GPT-2's transposed Conv1D."""
    if isinstance(base, nn.Linear):
        return base.in_features, base.out_features
    w = getattr(base, "weight", None)
    if w is None or w.dim() != 2:
        raise TypeError(f"cannot wrap {type(base).__name__} with LoRA")
    # transformers' Conv1D stores weight as (in, out) — the transpose of Linear.
    if type(base).__name__ == "Conv1D":
        return w.shape[0], w.shape[1]
    return w.shape[1], w.shape[0]


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Module, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        in_f, out_f = _shape_of(base)
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Linear(in_f, r, bias=False)
        self.lora_B = nn.Linear(r, out_f, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)          # start as an exact no-op
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = True

        dtype = next(base.parameters()).dtype
        if dtype in (torch.float16, torch.bfloat16):
            # Keep adapter maths in fp32 even under a half-precision base: the
            # updates are small and rank-r, and fp16 accumulation eats them.
            self.lora_A.to(torch.float32)
            self.lora_B.to(torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        h = self.lora_dropout(x).to(self.lora_A.weight.dtype)
        delta = self.lora_B(self.lora_A(h)) * self.scaling
        return out + delta.to(out.dtype)


    @torch.no_grad()
    def merge_(self) -> None:
        """Fold the adapter into the frozen weight, in place.

        ``delta = (alpha/r) · B·A`` has the shape of a Linear weight, ``(out, in)``.
        transformers' Conv1D stores its weight transposed, so it needs the
        transpose added instead — get this wrong and the export silently produces
        a scrambled model that still loads.
        """
        delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        w = self.base.weight
        w.add_((delta.t() if type(self.base).__name__ == "Conv1D" else delta).to(w.dtype))
        # Zero the adapter so a second merge cannot double-apply it.
        self.lora_B.weight.zero_()


def _matches(name: str, targets: list[str]) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    return leaf in targets


def inject(
    model: nn.Module,
    *,
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    targets: list[str] | None = None,
) -> dict[str, Any]:
    """Replace matching submodules in place. Returns a summary for the run log."""
    targets = targets or DEFAULT_TARGETS
    replaced: list[str] = []

    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if not _matches(full, targets):
                continue
            if isinstance(child, LoRALinear) or "lm_head" in full or "embed" in full:
                continue
            try:
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
                replaced.append(full)
            except TypeError:
                continue

    if not replaced:
        raise RuntimeError(
            "LoRA matched no modules. Set lora.targets explicitly for this architecture; "
            f"tried {targets}"
        )

    for n, p in model.named_parameters():
        p.requires_grad_("lora_A" in n or "lora_B" in n)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "modules_replaced": len(replaced),
        "targets_matched": sorted({m.rsplit(".", 1)[-1] for m in replaced}),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / max(total, 1), 4),
        "r": r, "alpha": alpha, "dropout": dropout,
    }


def lora_modules(model: nn.Module) -> list[LoRALinear]:
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


@contextlib.contextmanager
def adapters_disabled(model: nn.Module) -> Iterator[None]:
    """Temporarily run as the untouched base model.

    This *is* the reference policy for DPO, PPO, GRPO and GSPO. Because the
    adapters are additive and zero-initialised, switching them off recovers the
    original weights exactly — no second model, no drift between the reference we
    think we have and the one we compute against.
    """
    mods = lora_modules(model)
    previous = [m.enabled for m in mods]
    for m in mods:
        m.enabled = False
    try:
        yield
    finally:
        for m, was in zip(mods, previous):
            m.enabled = was


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def merge_and_unload(model: nn.Module) -> int:
    """Fold every adapter into its base weight and remove the wrappers.

    Afterwards the model is an ordinary transformers model with no trace of this
    system in it — which is the point. An export nobody needs SLM Foundry to load
    is an export that outlives it.
    """
    merged = 0
    for _, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                child.merge_()
                setattr(parent, child_name, child.base)
                merged += 1
    for p in model.parameters():
        p.requires_grad_(False)
    return merged


# ------------------------------------------------------------------ checkpoints

def save(model: nn.Module, out_dir: Path, meta: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()
             if "lora_A" in k or "lora_B" in k}
    torch.save(state, out_dir / "adapter.pt")
    (out_dir / "adapter_config.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def load(model: nn.Module, adapter_dir: str | Path) -> dict[str, Any]:
    """Load adapter weights onto an already-injected model.

    Strict about missing keys: a silent partial load produces a model that is
    *almost* the one the registry says it is, which is worse than an error.
    """
    d = Path(adapter_dir)
    meta = json.loads((d / "adapter_config.json").read_text())
    state = torch.load(d / "adapter.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    still_missing = [k for k in missing if "lora_" in k]
    if still_missing or unexpected:
        raise RuntimeError(
            f"adapter does not fit this model: {len(still_missing)} missing, "
            f"{len(unexpected)} unexpected LoRA tensors"
        )
    return meta
