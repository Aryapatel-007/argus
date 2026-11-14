"""Model health and circuit breakers.

Throughput has been measured collapsing ~5x mid-session under VRAM pressure
(CLAUDE.md). A model whose tok/s falls below `degrade_ratio` x its profiled
tok/s on `strikes` consecutive calls, or that times out `strikes` times in a
row, is marked degraded and the router stops choosing it.

Recovery is a half-open breaker: after `cooldown_s` the model is allowed ONE
trial call. That call is the health check. A healthy result closes the
breaker; a bad one re-opens it for another cooldown. State lives in
`model_health` so the dashboard shows it and a restart keeps it.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from argus.graph import registry
from argus.storage import db


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


class HealthMonitor:
    def __init__(self, conn: sqlite3.Connection, cfg: Dict, now=_now):
        self.conn = conn
        self.degrade_ratio = float(cfg.get("degrade_ratio", 0.35))
        self.strikes = int(cfg.get("strikes", 2))
        self.cooldown = timedelta(seconds=float(cfg.get("cooldown_s", 300)))
        self.now = now
        #: Models whose single trial call has been handed out (review M9: a
        #: half-open breaker allows ONE call, not every call after cooldown).
        self._trial_granted: set = set()

    def grant_trial(self, model: str) -> None:
        self._trial_granted.add(model)

    def routable(self, model: str) -> str:
        """status() as the router should see it: a trial already granted
        reads as degraded until its result is recorded."""
        s = self.status(model)
        return "degraded" if s == "trial" and model in self._trial_granted else s

    def _row(self, model: str) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM model_health WHERE model_tag = ?", (model,)).fetchone()

    def status(self, model: str) -> str:
        """'healthy', 'degraded', or 'trial' (cooldown over: one call allowed)."""
        row = self._row(model)
        if row is None or row["status"] == "healthy":
            return "healthy"
        degraded_at = datetime.fromisoformat(row["degraded_at"].replace("Z", "+00:00"))
        return "trial" if self.now() - degraded_at >= self.cooldown else "degraded"

    def record(self, model: str, *, tok_s: Optional[float], timed_out: bool, ok: bool,
               responded: bool = True) -> str:
        """`responded=False`: the call got no response at all (model not
        pulled, server error). That counts as a strike like a timeout (review
        M9). A schema-invalid answer is a quality problem, not a health one."""
        self._trial_granted.discard(model)
        baseline = registry.profile(self.conn, model, "*", "tok_s")
        row = self._row(model)
        timeouts = (row["consecutive_timeouts"] if row else 0)
        slow = (row["consecutive_slow"] if row else 0)
        was = self.status(model)
        is_slow = bool(baseline and tok_s is not None and tok_s < baseline * self.degrade_ratio)
        hard_fail = timed_out or not responded
        timed_out = hard_fail
        timeouts = timeouts + 1 if hard_fail else 0
        slow = slow + 1 if is_slow else 0
        bad_trial = was == "trial" and (timed_out or is_slow)
        if timeouts >= self.strikes or slow >= self.strikes or bad_trial:
            status, reason = "degraded", (
                f"{timeouts} consecutive timeouts or failed calls" if timeouts >= self.strikes else
                f"tok/s {tok_s:.1f} < {self.degrade_ratio:.0%} of profiled {baseline:.1f}"
                if is_slow and baseline else "failed its trial call")
            degraded_at = _fmt(self.now())
        elif was in ("degraded",):
            status, reason, degraded_at = "degraded", row["reason"], row["degraded_at"]
        else:
            status, reason, degraded_at = "healthy", None, None
        with db.atomic(self.conn):
            self.conn.execute(
                "INSERT INTO model_health (model_tag, status, reason, recent_tok_s, baseline_tok_s, "
                "consecutive_timeouts, consecutive_slow, degraded_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (model_tag) DO UPDATE SET "
                "status = excluded.status, reason = excluded.reason, recent_tok_s = excluded.recent_tok_s, "
                "baseline_tok_s = excluded.baseline_tok_s, consecutive_timeouts = excluded.consecutive_timeouts, "
                "consecutive_slow = excluded.consecutive_slow, degraded_at = excluded.degraded_at, "
                "updated_at = excluded.updated_at",
                (model, status, reason, tok_s, baseline, timeouts, slow, degraded_at, _fmt(self.now())),
            )
        return status
