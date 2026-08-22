"""Where things live on disk.

Artifact layout, rooted at ``FOUNDRY_DATA_DIR`` (default ``<repo>/var``):

    var/
      uploads/<team>/<dataset_id>/data.jsonl     immutable once validated
      runs/<run_id>/
        config.json                              exact resolved config used
        adapter/                                 LoRA weights + tokenizer refs
        metrics.jsonl                            one JSON object per logged step
        samples.jsonl                            generations captured during RL
      evals/<eval_id>/per_example.jsonl
      models/<model_version_id>/                 promoted, read-only copy

The split between ``runs/`` and ``models/`` is deliberate. A run directory is
scratch that a retry may overwrite; a model-version directory is a copy taken at
promotion time and is never written again. Registry rows point at the latter, so
re-running a job cannot silently change what a promoted model version *is*.
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]          # slm-foundry/
REPO_ROOT = PROJECT_ROOT.parent                  # the workspace root


def data_dir() -> Path:
    p = Path(os.environ.get("FOUNDRY_DATA_DIR", PROJECT_ROOT / "var"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sub(*parts: str) -> Path:
    p = data_dir().joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def upload_dir(team_slug: str, dataset_id: int) -> Path:
    return _sub("uploads", team_slug, str(dataset_id))


def run_dir(run_id: int) -> Path:
    return _sub("runs", str(run_id))


def eval_dir(eval_id: int) -> Path:
    return _sub("evals", str(eval_id))


def model_dir(model_version_id: int) -> Path:
    return _sub("models", str(model_version_id))


def static_dir() -> Path:
    return PROJECT_ROOT / "static"


def config_path() -> Path:
    return Path(os.environ.get("FOUNDRY_CONFIG", PROJECT_ROOT / "configs" / "foundry.json"))
