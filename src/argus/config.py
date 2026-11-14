"""Config loader: config/targets.yaml (what to watch) + config/models.yaml
(which models are allowed, the named baseline, the embedder, Ollama options).

Model tags live in config and are NEVER hardcoded in source (CLAUDE.md).
There are no capability tiers: what a model is good at is measured
(model_profiles) and learned (arm_stats), not configured (ARGUS_BUILD 6.1).

`load_config()` returns one merged dict: the targets file's keys plus
`models`, `allowed`, `ollama` from models.yaml. Set ARGUS_CONFIG_DIR to point
the whole thing at another directory (tests, eval runs).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

# src/argus/config.py -> parents[2] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"
# Kept for callers that still name the targets file directly.
CONFIG_PATH = DEFAULT_CONFIG_DIR / "targets.yaml"

#: Legacy linear-loop roles -> models.yaml `models:` key. The graph engine
#: does not use this; its router picks per node from the allowlist.
DEFAULT_ROLE_MODEL_KEYS = {
    "planner": "baseline",
    "critic": "baseline",
    "writer": "baseline",
    "judge": "baseline",
    # The one role that is NOT a chat model.
    "embedder": "embed",
}


def config_dir() -> Path:
    return Path(os.environ.get("ARGUS_CONFIG_DIR") or DEFAULT_CONFIG_DIR)


@functools.lru_cache(maxsize=8)
def _load(directory: str) -> Dict[str, Any]:
    d = Path(directory)
    with open(d / "targets.yaml", "r", encoding="utf-8") as handle:
        merged: Dict[str, Any] = yaml.safe_load(handle) or {}
    models_path = d / "models.yaml"
    if models_path.exists():
        with open(models_path, "r", encoding="utf-8") as handle:
            models = yaml.safe_load(handle) or {}
        for key in ("models", "allowed", "ollama", "roles"):
            if key in models:
                merged[key] = models[key]
    return merged


def load_config(config_path: Any = None) -> Dict[str, Any]:
    """Merged config. `config_path` may be a directory or a targets.yaml path."""
    if config_path is None:
        directory = config_dir()
    else:
        p = Path(config_path)
        directory = p if p.is_dir() else p.parent
    return _load(str(directory))


def clear_cache() -> None:
    _load.cache_clear()


def normalize_tag(tag: str) -> str:
    """`phi4-mini` and `phi4-mini:latest` name the same Ollama model."""
    return tag if ":" in tag else f"{tag}:latest"


def allowed_models(config_path: Any = None) -> List[str]:
    return [normalize_tag(t) for t in (load_config(config_path).get("allowed") or [])]


def model_for_role(role: str, config_path: Any = None) -> str:
    """Resolve a legacy role (or the embedder) to its concrete model tag."""
    cfg = load_config(config_path)
    overrides = cfg.get("roles") or {}
    key = overrides.get(role) or DEFAULT_ROLE_MODEL_KEYS.get(role)
    if key is None:
        raise ValueError(f"no model binding configured for role {role!r}")
    models = cfg.get("models") or {}
    if key in models:
        return models[key]
    if normalize_tag(key) in allowed_models(config_path):
        return key  # an override naming an allowed tag directly
    raise ValueError(
        f"role {role!r} resolves to {key!r}, which is neither a models: key "
        f"({sorted(models)}) nor an allowed tag ({allowed_models(config_path)})"
    )


def ollama_options(config_path: Any = None) -> Dict[str, Any]:
    return dict(load_config(config_path).get("ollama") or {})


def targets(config_path: Any = None) -> List[Dict[str, Any]]:
    return list(load_config(config_path).get("targets") or [])
