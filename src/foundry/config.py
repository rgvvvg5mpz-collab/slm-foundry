"""Configuration loading and the method catalogue.

The catalogue in ``configs/foundry.json`` is read by three different consumers —
the run wizard (to draw its forms), the API (to validate a submitted config), and
the trainers (to fill defaults). They must agree, so they all read this module
rather than each carrying its own copy of "what is a valid DPO beta".
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

from .paths import config_path


@lru_cache(maxsize=1)
def cfg() -> dict[str, Any]:
    return json.loads(config_path().read_text())


def reload() -> None:
    cfg.cache_clear()


# ------------------------------------------------------------------ environment

def database_url() -> str:
    return os.environ.get("FOUNDRY_DATABASE_URL", "sqlite:///./foundry.db")


def anthropic_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY")


def execution_backend() -> str:
    """``auto`` | ``torch`` | ``tiny``.

    ``tiny`` runs the *same* objective code against a randomly-initialised
    two-layer model with a vocabulary built from the uploaded data, so all seven
    pipelines execute for real in seconds on a laptop with nothing downloaded.
    ``auto`` picks ``torch`` when the requested base model is present locally and
    falls back to ``tiny`` otherwise. See :mod:`foundry.trainers.tiny`.
    """
    return os.environ.get("FOUNDRY_BACKEND", "auto").lower()


# -------------------------------------------------------------------- catalogue

def methods() -> dict[str, Any]:
    return cfg()["methods"]


def method_spec(name: str) -> dict[str, Any]:
    m = methods().get(name)
    if m is None:
        raise KeyError(f"unknown method: {name}")
    return m


def method_names() -> list[str]:
    return list(methods())


def base_models() -> list[dict[str, Any]]:
    return cfg()["base_models"]


def base_model_ids() -> set[str]:
    return {b["id"] for b in base_models()}


def queue_config() -> dict[str, Any]:
    return cfg()["queue"]


def limits() -> dict[str, Any]:
    return cfg()["limits"]


def judge_config() -> dict[str, Any]:
    return cfg()["judge"]


# -------------------------------------------------------------------- validation

def defaults_for(method: str) -> dict[str, Any]:
    return {p["name"]: p["default"] for p in method_spec(method)["params"]}


_COERCE = {
    "int": int,
    "float": float,
    "bool": bool,
    "enum": str,
    "string": str,
}


def validate_params(method: str, submitted: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Coerce and range-check a submitted hyperparameter block.

    Returns ``(resolved, errors)``. Unknown keys are dropped rather than rejected:
    a stale browser tab holding a parameter we have since removed should not be
    able to fail a run submission, and silently ignoring it is the behaviour that
    keeps the resolved config honest about what was actually used.
    """
    spec = method_spec(method)
    resolved: dict[str, Any] = {}
    errors: list[str] = []

    for p in spec["params"]:
        name, ptype = p["name"], p["type"]
        raw = submitted.get(name, p["default"])
        try:
            if ptype == "bool":
                value = raw if isinstance(raw, bool) else str(raw).lower() in ("1", "true", "yes", "on")
            else:
                value = _COERCE[ptype](raw)
        except (TypeError, ValueError):
            errors.append(f"{name}: expected {ptype}, got {raw!r}")
            resolved[name] = p["default"]
            continue

        if ptype in ("int", "float"):
            lo, hi = p.get("min"), p.get("max")
            if lo is not None and value < lo:
                errors.append(f"{name}: {value} below minimum {lo}")
                value = lo
            if hi is not None and value > hi:
                errors.append(f"{name}: {value} above maximum {hi}")
                value = hi
        elif ptype == "enum":
            allowed = {o["value"] for o in p.get("options", [])}
            if allowed and value not in allowed:
                errors.append(f"{name}: {value!r} not one of {sorted(allowed)}")
                value = p["default"]

        resolved[name] = value

    return resolved, errors
