"""`argus` command line.

    argus migrate                      apply migrations to data/argus.db
    argus watch [--target ID ...]      one run per target (all targets by default)
    argus resume RUN_ID                drive an unfinished run on
    argus resume-decided               continue every run whose approval a human decided
    argus approvals [list|approve|reject] ...
    argus runs | trace RUN_ID | why NODE_ID | health
    argus reflect TARGET_ID            consolidate a target's facts (memory reflection)
    argus dashboard [--port 8787]      local web dashboard on 127.0.0.1
    argus scheduler                    run targets on their schedule (APScheduler, in process)
    argus up                           dashboard + scheduler in one process
    argus run-legacy ...               the Sprint 1 linear loop

Common: --db PATH, --mode baseline|fixed|learning.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import sys
from pathlib import Path
from typing import List, Optional

from argus import app, config, observe
from argus.storage import db


def _opts(args) -> app.Options:
    return app.Options(db_path=Path(args.db), mode=getattr(args, "mode", "fixed"))


def _print_outcomes(outs) -> int:
    worst = 0
    for o in outs:
        print(f"{o.run_id or '-':20} {o.status:18} {(o.digest or '(no changes)')[:200]}")
        if o.status == "failed" or str(o.status).startswith("error"):
            worst = 1
    return worst


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="argus")
    ap.add_argument("--db", default=str(db.DB_PATH))
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("migrate")
    w = sub.add_parser("watch")
    w.add_argument("--target", nargs="*")
    w.add_argument("--mode", default="fixed", choices=["baseline", "fixed", "learning"])
    r = sub.add_parser("resume")
    r.add_argument("run_id")
    r.add_argument("--mode", default="fixed", choices=["baseline", "fixed", "learning"])
    rd = sub.add_parser("resume-decided")
    rd.add_argument("--mode", default="fixed", choices=["baseline", "fixed", "learning"])
    a = sub.add_parser("approvals")
    a.add_argument("action", nargs="?", default="list", choices=["list", "approve", "reject", "all"])
    a.add_argument("approval_id", nargs="?", type=int)
    a.add_argument("--note", default="")
    a.add_argument("--then-resume", action="store_true", help="continue the waiting run right away")
    sub.add_parser("runs")
    t = sub.add_parser("trace")
    t.add_argument("run_id")
    y = sub.add_parser("why")
    y.add_argument("node_id")
    sub.add_parser("health")
    rf = sub.add_parser("reflect")
    rf.add_argument("target_id")
    d = sub.add_parser("dashboard")
    d.add_argument("--port", type=int, default=8787)
    s = sub.add_parser("scheduler")
    s.add_argument("--mode", default="fixed", choices=["baseline", "fixed", "learning"])
    # Test and demo switches (never needed in production):
    s.add_argument("--interval-s", type=float, help="run every target every N seconds")
    s.add_argument("--fake-backend", action="store_true")
    s.add_argument("--fake-embed", action="store_true")
    s.add_argument("--delay", type=float, default=0.0, help="seconds per fake model call")
    s.add_argument("--pages", help="JSON {url: page} instead of the browser")
    s.add_argument("--models", nargs="*")
    u = sub.add_parser("up")
    u.add_argument("--port", type=int, default=8787)
    u.add_argument("--mode", default="fixed", choices=["baseline", "fixed", "learning"])
    legacy = sub.add_parser("run-legacy")
    legacy.add_argument("rest", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    db_path = Path(args.db)

    if args.command == "migrate":
        from argus.storage.migrate import apply_migrations
        for f in apply_migrations(db_path):
            print(f"applied {f}")
        print(f"database: {db_path}")
        return 0
    if args.command == "watch":
        return _print_outcomes(asyncio.run(app.watch_all(_opts(args), args.target)))
    if args.command == "resume":
        async def go():
            async with app.Session(_opts(args)) as session:
                return await session.resume(args.run_id)
        return _print_outcomes(asyncio.run(go()))
    if args.command == "resume-decided":
        return _print_outcomes(asyncio.run(app.resume_decided(_opts(args))))
    if args.command == "approvals":
        if args.action in ("list", "all"):
            rows = app.list_approvals(db_path, None if args.action == "all" else "pending")
            if not rows:
                print("no pending approvals")
            for row in rows:
                action = json.loads(row["proposed_action"])
                print(f"#{row['approval_id']} [{row['status']}] {row['target_id']} {row['action_kind']}: "
                      f"{action.get('subject', '')}\n    {action.get('body', '')[:300]}\n"
                      f"    why: {row['reasoning']}")
            return 0
        if args.approval_id is None:
            print("approval id required", file=sys.stderr)
            return 2
        try:
            row = app.decide(db_path, args.approval_id,
                             "approved" if args.action == "approve" else "rejected",
                             by=f"cli:{getpass.getuser()}", note=args.note)
        except app.DecisionError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"#{row['approval_id']} is now {row['status']}")
        if args.then_resume:
            return _print_outcomes(asyncio.run(app.resume_decided(app.Options(db_path=db_path))))
        print("run `argus resume-decided` (or the scheduler will) to continue the run")
        return 0
    if args.command == "runs":
        for row in observe.runs(db_path):
            print(f"{row['started_at']} {row['run_id']:18} {row['target_id']:26} {row['status']:10} "
                  f"graph={row['graph_status'] or '-':16} nodes={row['nodes'] or 0} calls={row['calls'] or 0}")
        return 0
    if args.command == "trace":
        tr = observe.trace(db_path, args.run_id)
        for g in tr["graphs"]:
            print(f"graph {g['graph_id']} [{g['status']}] swaps={g['swaps']} tokens={g['tokens_used']}")
            for sp in g["spans"]:
                model = f" {sp['model']}/{sp['strategy']}" if sp["model"] else ""
                note = sp["error"] or sp["skip_reason"] or ""
                print(f"  {sp['type']:15} a{sp['attempt']} {sp['status']:8} {sp['duration_s'] or 0:7.2f}s"
                      f"{model} {note[:90]}")
        return 0
    if args.command == "why":
        for dcs in observe.why(db_path, args.node_id)["decisions"]:
            flag = " (superseded)" if dcs["superseded"] else ""
            print(f"{dcs['created_at']} -> {dcs['model_tag']} [{dcs['strategy']}, {dcs['mode']}]{flag}: {dcs['reason']}")
            for c in dcs["candidates"]:
                print(f"    {c.get('model'):24} utility={c.get('utility')} q={c.get('quality')} "
                      f"lat={c.get('latency_s')} swap={c.get('swap_s')} risk={c.get('risk')} "
                      f"{c.get('excluded') or ''}")
        return 0
    if args.command == "health":
        for h in observe.health(db_path):
            print(f"{h['model']:24} {h['status']:9} recent tok/s={h['recent_tok_s']} "
                  f"profiled tok/s={h['profile'].get('tok_s')} {h['reason'] or ''}")
        return 0
    if args.command == "reflect":
        target = next((t for t in config.targets() if t["id"] == args.target_id), {"id": args.target_id})
        async def go():
            async with app.Session(app.Options(db_path=db_path)) as session:
                run_id, gid = session.engine.start_run(target, kind="reflect")
                await session.memory.open(run_id)
                try:
                    return [await session.engine.run_graph(gid)]
                finally:
                    await session.memory.close()
        return _print_outcomes(asyncio.run(go()))
    if args.command == "dashboard":
        from argus.dashboard import serve
        serve(db_path, port=args.port)
        return 0
    if args.command == "scheduler":
        from argus.scheduler import run_forever
        from argus.graph.fakes import FakeBackend
        run_forever(app.Options(
            db_path=db_path, mode=args.mode, models=args.models, fake_embed=args.fake_embed,
            backend=FakeBackend(delay_s=args.delay) if args.fake_backend else None,
            pages=json.loads(Path(args.pages).read_text(encoding="utf-8")) if args.pages else None,
        ), interval_s=args.interval_s)
        return 0
    if args.command == "up":
        from argus.scheduler import run_forever_with_dashboard
        run_forever_with_dashboard(app.Options(db_path=db_path, mode=args.mode), port=args.port)
        return 0
    if args.command == "run-legacy":
        from argus.orchestrator import runner
        return runner.main(args.rest)
    return 2


if __name__ == "__main__":
    sys.exit(main())
