"""Model registry: which models exist, which are allowed, what was measured.

    installed (ollama list)  x  allowed (config/models.yaml)  x  profiles (DB)

The pool is never hardcoded. A model that is allowed but not installed is
reported and left out; a model that is installed but not allowed is never
used. What a model is good at comes only from `model_profiles` (profiler
priors) and `arm_stats` (learned), never from config.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from argus import config
from argus.ollama_client import get_client


@dataclass
class ModelInfo:
    tag: str
    installed: bool
    size_bytes: Optional[int] = None
    family: Optional[str] = None
    parameter_size: Optional[str] = None
    quantization: Optional[str] = None


def installed_models(host: Optional[str] = None) -> Dict[str, ModelInfo]:
    client = get_client(host or config.ollama_options().get("host"), 30)
    out: Dict[str, ModelInfo] = {}
    for m in client.list().models or []:
        tag = config.normalize_tag(m.model)
        details = getattr(m, "details", None)
        out[tag] = ModelInfo(
            tag=tag, installed=True, size_bytes=getattr(m, "size", None),
            family=getattr(details, "family", None) if details else None,
            parameter_size=getattr(details, "parameter_size", None) if details else None,
            quantization=getattr(details, "quantization_level", None) if details else None,
        )
    return out


def pool(installed: Optional[Dict[str, ModelInfo]] = None) -> Tuple[List[ModelInfo], List[str]]:
    """(usable chat models, allowed-but-missing tags)."""
    installed = installed if installed is not None else installed_models()
    usable, missing = [], []
    for tag in config.allowed_models():
        if tag in installed:
            usable.append(installed[tag])
        else:
            missing.append(tag)
    return usable, missing


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------


def profiles(conn: sqlite3.Connection) -> Dict[Tuple[str, str, str], float]:
    """(model, node_type, metric) -> value. node_type '*' = model-wide."""
    return {
        (r["model_tag"], r["node_type"], r["metric"]): r["value"]
        for r in conn.execute("SELECT model_tag, node_type, metric, value FROM model_profiles")
        if r["value"] is not None
    }


def profile(conn: sqlite3.Connection, model: str, node_type: str, metric: str,
            default: Optional[float] = None) -> Optional[float]:
    r = conn.execute(
        "SELECT value FROM model_profiles WHERE model_tag = ? AND node_type = ? AND metric = ?",
        (model, node_type, metric),
    ).fetchone()
    return r["value"] if r is not None and r["value"] is not None else default


def coresidency(conn: sqlite3.Connection) -> Dict[Tuple[str, str], bool]:
    out = {}
    for r in conn.execute("SELECT model_a, model_b, coresident FROM co_residency"):
        out[(r["model_a"], r["model_b"])] = bool(r["coresident"])
        out[(r["model_b"], r["model_a"])] = bool(r["coresident"])
    return out


def write_profile(conn: sqlite3.Connection, model: str, node_type: str, metric: str,
                  value: Optional[float], samples: int = 0, detail: Optional[str] = None,
                  profile_run: Optional[str] = None) -> None:
    """Caller holds the transaction (db.atomic)."""
    conn.execute(
        "INSERT INTO model_profiles (model_tag, node_type, metric, value, samples, detail, profile_run) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (model_tag, node_type, metric) DO UPDATE SET value = excluded.value, "
        "samples = excluded.samples, detail = excluded.detail, profile_run = excluded.profile_run, "
        "measured_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
        (model, node_type, metric, value, samples, detail, profile_run),
    )


def write_coresidency(conn: sqlite3.Connection, a: str, b: str, coresident: bool,
                      detail: Optional[str] = None) -> None:
    a, b = sorted((a, b))
    conn.execute(
        "INSERT INTO co_residency (model_a, model_b, coresident, detail) VALUES (?, ?, ?, ?) "
        "ON CONFLICT (model_a, model_b) DO UPDATE SET coresident = excluded.coresident, "
        "detail = excluded.detail, measured_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
        (a, b, int(coresident), detail),
    )
