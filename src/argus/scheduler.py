"""In-process scheduler (APScheduler 3.x, AsyncIOScheduler).

Jobs:
  - watch:<target>     the target's `frequency` (daily | weekly), or a
                       `schedule` cron string in targets.yaml. One run at a time
                       per target (max_instances=1), missed runs coalesce.
  - catch-up           at start, any target whose last watch is older than its
                       period runs immediately (a laptop that was off).
  - recover            at start and every 10 minutes: runs a hard kill left
                       half-done (graph running, lease free or owner dead)
                       are resumed; runs whose approvals a human decided are
                       continued.
  - reflect:<target>   memory consolidation policy: weekly, and only when the
                       target gained at least `min_new_facts` fact versions
                       since its last reflection.

Everything is serialised through one asyncio lock, so the scheduler never
runs two graphs at once on a GPU that fits one model (ARGUS_BUILD 6.2).
"""

from __future__ import annotations

import asyncio
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from argus import app, config
from argus.storage import db, lease

PERIOD = {"daily": timedelta(days=1), "weekly": timedelta(days=7)}
CRON = {"daily": {"hour": 7, "minute": 0}, "weekly": {"day_of_week": "mon", "hour": 7, "minute": 0}}
MIN_NEW_FACTS_FOR_REFLECT = 2


def log(msg: str) -> None:
    print(f"[argus-scheduler {datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def interrupted_runs(db_path: Path) -> List[str]:
    """Runs with a graph still 'running' whose lease is gone or whose owner
    process is dead: what a hard kill leaves behind."""
    out = []
    with db.connect(db_path) as conn:
        for r in conn.execute(
                "SELECT DISTINCT g.run_id, l.owner FROM graphs g LEFT JOIN run_leases l ON l.run_id = g.run_id "
                "WHERE g.status = 'running'"):
            if r["owner"] is None or lease._owner_liveness(r["owner"]) is False:
                out.append(r["run_id"])
    return out


def last_watch(db_path: Path, target_id: str) -> Optional[datetime]:
    with db.connect(db_path) as conn:
        r = conn.execute("SELECT MAX(started_at) AS t FROM runs r JOIN graphs g ON g.run_id = r.run_id "
                         "WHERE r.target_id = ? AND g.kind = 'watch'", (target_id,)).fetchone()
    return datetime.fromisoformat(r["t"].replace("Z", "+00:00")) if r and r["t"] else None


def new_facts_since_reflect(db_path: Path, target_id: str) -> int:
    with db.connect(db_path) as conn:
        last = conn.execute("SELECT MAX(r.started_at) AS t FROM runs r JOIN graphs g ON g.run_id = r.run_id "
                            "WHERE r.target_id = ? AND g.kind = 'reflect'", (target_id,)).fetchone()["t"]
        return conn.execute(
            "SELECT COUNT(*) FROM facts WHERE target_id = ? AND entity_key NOT LIKE 'summary:%' "
            "AND change_kind IN ('added', 'modified') AND created_at > ?",
            (target_id, last or "0000")).fetchone()[0]


class Scheduler:
    def __init__(self, opts: app.Options, *, interval_s: Optional[float] = None):
        self.opts = opts
        self.interval_s = interval_s  # tests: every target every N seconds
        self.lock = asyncio.Lock()
        self.sched = None

    async def watch(self, target: Dict[str, Any]) -> None:
        async with self.lock:
            log(f"watch {target['id']}")
            async with app.Session(self.opts) as s:
                out = await s.run_target(target)
            log(f"watch {target['id']}: {out.status} {(out.digest or '')[:120]}")

    async def recover(self) -> None:
        async with self.lock:
            runs = interrupted_runs(Path(self.opts.db_path)) + app.decided_waiting_runs(Path(self.opts.db_path))
            if not runs:
                return
            async with app.Session(self.opts) as s:
                for run_id in dict.fromkeys(runs):
                    try:
                        outs = await s.resume(run_id)
                        log(f"resumed {run_id}: {[o.status for o in outs]}")
                    except lease.LeaseLost:
                        log(f"{run_id} is owned by a live process; left alone")

    async def reflect(self, target: Dict[str, Any]) -> None:
        if new_facts_since_reflect(Path(self.opts.db_path), target["id"]) < MIN_NEW_FACTS_FOR_REFLECT:
            return
        async with self.lock:
            async with app.Session(self.opts) as s:
                run_id, gid = s.engine.start_run(target, kind="reflect")
                await s.memory.open(run_id)
                try:
                    out = await s.engine.run_graph(gid)
                finally:
                    await s.memory.close()
            log(f"reflect {target['id']}: {out.status}")

    def start(self) -> None:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.interval import IntervalTrigger

        from argus.storage.migrate import apply_migrations

        apply_migrations(Path(self.opts.db_path))  # the scheduler reads the db before any run
        self.sched = AsyncIOScheduler(timezone="UTC")
        now = datetime.now(timezone.utc)
        for t in config.targets():
            if self.interval_s:
                trigger = IntervalTrigger(seconds=self.interval_s)
            elif t.get("schedule"):
                trigger = CronTrigger.from_crontab(t["schedule"])
            else:
                trigger = CronTrigger(**CRON.get(t.get("frequency", "weekly"), CRON["weekly"]))
            self.sched.add_job(self.watch, trigger, args=[t], id=f"watch:{t['id']}", max_instances=1,
                               coalesce=True, misfire_grace_time=6 * 3600)
            last = last_watch(Path(self.opts.db_path), t["id"])
            if self.interval_s or last is None or now - last > PERIOD.get(t.get("frequency"), PERIOD["weekly"]):
                self.sched.add_job(self.watch, args=[t], id=f"catchup:{t['id']}")  # run now
            self.sched.add_job(self.reflect, CronTrigger(day_of_week="sun", hour=8), args=[t],
                               id=f"reflect:{t['id']}", max_instances=1, coalesce=True)
        self.sched.add_job(self.recover, IntervalTrigger(minutes=10), id="recover", max_instances=1,
                           coalesce=True, next_run_time=now)  # also right now: resume what a kill left
        self.sched.start()
        log(f"started: {[j.id for j in self.sched.get_jobs()]}")


def run_forever(opts: app.Options, interval_s: Optional[float] = None) -> None:
    async def main():
        s = Scheduler(opts, interval_s=interval_s)
        s.start()
        stop = asyncio.Event()
        try:
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop.set)
        except (NotImplementedError, RuntimeError):
            pass  # Windows: Ctrl+C / process kill
        await stop.wait()
    asyncio.run(main())


def run_forever_with_dashboard(opts: app.Options, port: int = 8787) -> None:
    import threading

    from argus.dashboard import serve

    threading.Thread(target=serve, args=(Path(opts.db_path),), kwargs={"port": port}, daemon=True).start()
    run_forever(opts)
