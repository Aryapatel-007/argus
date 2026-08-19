"""Loader for config/targets.yaml.

Model tags live here and are NEVER hardcoded in source (CLAUDE.md convention),
so routing a role to a different model later is a config edit, not a refactor.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict

import yaml

# src/argus/config.py -> parents[2] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "targets.yaml"

#: Which config key under `models:` each role resolves to. Sprint 1 points every
#: chat role at the same model — zero swaps, per CLAUDE.md. An optional `roles:`
#: section in targets.yaml overrides this per role without a code change.
DEFAULT_ROLE_MODEL_KEYS = {
    "planner": "chat",
    "critic": "chat",
    "writer": "chat",
    "judge": "chat",
    "router": "fast",
}


@functools.lru_cache(maxsize=8)
def load_config(config_path: str | Path = CONFIG_PATH) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def model_for_role(role: str, config_path: str | Path = CONFIG_PATH) -> str:
    """Resolve a role to its concrete model tag from config."""
    config = load_config(config_path)
    key = config.get("roles", {}).get(role) or DEFAULT_ROLE_MODEL_KEYS.get(role)
    if key is None:
        raise ValueError(f"no model binding configured for role {role!r}")

    models = config.get("models", {})
    # Allow `roles:` to name a model tag directly as well as a models: key.
    tag = models.get(key, key if key in models.values() else None)
    if not tag:
        raise ValueError(f"config has no models entry for {key!r} (role {role!r})")
    return tag


def ollama_options(config_path: str | Path = CONFIG_PATH) -> Dict[str, Any]:
    return dict(load_config(config_path).get("ollama", {}))
