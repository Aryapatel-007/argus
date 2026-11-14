"""Learning: one Beta posterior per arm (node type, model). Thompson sampling.

Why Thompson over UCB (D-017): rewards here are bounded in [0, 1] and few
(a weekly system sees tens of calls per arm per month). Thompson sampling
uses the profiler's quality score directly as a Beta prior and degrades
gracefully with little data; UCB's confidence bonus is tuned for many pulls
and over-explores at this volume. Thompson also gives a natural greedy
fallback (the posterior mean) for the bounded-exploration rule.

Rewards are idempotent: `arm_rewards` has UNIQUE(node_id, model_tag, source)
and the posterior is updated in the SAME transaction only when the insert
actually added a row. A resumed run that re-reports a reward changes nothing.
"""

from __future__ import annotations

import random
import sqlite3
from typing import Optional, Tuple

from argus.storage import db


def seed_params(prior_quality: float, strength: float) -> Tuple[float, float]:
    q = min(max(prior_quality, 0.01), 0.99)
    return 1.0 + strength * q, 1.0 + strength * (1.0 - q)


def ensure_arm(conn: sqlite3.Connection, node_type: str, model: str,
               prior_quality: float, strength: float) -> Tuple[float, float]:
    row = conn.execute(
        "SELECT alpha, beta FROM arm_stats WHERE node_type = ? AND model_tag = ?", (node_type, model)
    ).fetchone()
    if row is not None:
        return row["alpha"], row["beta"]
    alpha, beta = seed_params(prior_quality, strength)
    with db.atomic(conn):
        conn.execute(
            "INSERT INTO arm_stats (node_type, model_tag, alpha, beta) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (node_type, model_tag) DO NOTHING",
            (node_type, model, alpha, beta),
        )
    row = conn.execute(
        "SELECT alpha, beta FROM arm_stats WHERE node_type = ? AND model_tag = ?", (node_type, model)
    ).fetchone()
    return row["alpha"], row["beta"]


def mean(alpha: float, beta: float) -> float:
    return alpha / (alpha + beta)


def sample(alpha: float, beta: float, rng: random.Random) -> float:
    return rng.betavariate(alpha, beta)


def apply_reward(conn: sqlite3.Connection, *, node_type: str, model: str, node_id: str,
                 source: str, reward: float, detail: Optional[str] = None,
                 prior_quality: float = 0.5, strength: float = 6.0) -> bool:
    """Record a reward once and update the arm. True if this call applied it."""
    reward = min(max(float(reward), 0.0), 1.0)
    ensure_arm(conn, node_type, model, prior_quality, strength)
    with db.atomic(conn):
        cur = conn.execute(
            "INSERT INTO arm_rewards (node_type, model_tag, node_id, source, reward, detail) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (node_id, model_tag, source) DO NOTHING",
            (node_type, model, node_id, source, reward, detail),
        )
        if cur.rowcount != 1:
            return False
        conn.execute(
            "UPDATE arm_stats SET alpha = alpha + ?, beta = beta + ?, pulls = pulls + 1, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE node_type = ? AND model_tag = ?",
            (reward, 1.0 - reward, node_type, model),
        )
    return True
