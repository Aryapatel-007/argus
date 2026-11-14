"""CLI for the graph engine.

    python -m argus.graph.runner --target monash_ai_masters
    python -m argus.graph.runner --resume <run_id>
    python -m argus.graph.runner --target-json t.json --pages pages.json --fake-backend --fake-embed

Exit codes: 0 done, 1 failed, 2 bad arguments, 3 run owned by a live
process, 4 waiting for approval (a normal outcome, not an error).

Test switches (crash tests, offline runs): --fake-backend (deterministic
model, no Ollama), --fake-embed (deterministic vectors), --pages (serve pages
from a JSON file instead of the browser), --delay (seconds per fake model
call, to widen crash windows), --hang ROLE:INDEX (the fake hangs on that
call so a crash test can kill the process while it is in flight).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from argus import config
from argus.graph import fakes
from argus import app
from argus.graph.services import ActionExecutor
from argus.storage import db, lease

EXIT = {"done": 0, "failed": 1, "waiting_approval": 4}


class _HangingExecutor(ActionExecutor):
    """Crash tests only. `after_claim`: hang before the effect (the approval
    row is already claimed). `after_effect`: write the outbox file, then
    hang before the node's done is recorded."""

    def __init__(self, outbox: Path, when: str):
        super().__init__(outbox)
        self.when = when

    def execute(self, kind, payload, key):
        import time

        if self.when == "after_claim":
            time.sleep(3600)
        result = super().execute(kind, payload, key)
        if self.when == "after_effect":
            time.sleep(3600)
        return result


def _target(args) -> Dict[str, Any]:
    if args.target_json:
        return json.loads(Path(args.target_json).read_text(encoding="utf-8"))
    for t in config.targets():
        if t["id"] == args.target:
            return t
    raise SystemExit(f"unknown target {args.target!r}; known: {[t['id'] for t in config.targets()]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="argus.graph.runner")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--target")
    g.add_argument("--target-json")
    g.add_argument("--resume", metavar="RUN_ID")
    ap.add_argument("--db", default=str(db.DB_PATH))
    ap.add_argument("--mode", default="fixed", choices=["baseline", "fixed", "learning"])
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--pages")
    ap.add_argument("--outbox")
    ap.add_argument("--fake-backend", action="store_true")
    ap.add_argument("--fake-embed", action="store_true")
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--hang")
    ap.add_argument("--hang-remember-after", type=int)
    ap.add_argument("--hang-execute", choices=["after_claim", "after_effect"])
    ap.add_argument("--run-id")
    ap.add_argument("--memory", default="mcp", choices=["mcp", "direct"],
                    help="mcp: memory server over stdio (production); direct: in-process (tests)")
    args = ap.parse_args(argv)

    db_path = Path(args.db)
    backend = None
    if args.fake_backend:
        backend = fakes.FakeBackend(delay_s=args.delay)
        if args.hang:
            role, index = args.hang.split(":")
            backend.hang_on = (role, int(index))
    outbox = Path(args.outbox) if args.outbox else db_path.parent / "outbox"
    opts = app.Options(
        db_path=db_path, mode=args.mode, models=args.models, memory=args.memory,
        pages=json.loads(Path(args.pages).read_text(encoding="utf-8")) if args.pages else None,
        fake_embed=args.fake_embed, outbox=outbox, backend=backend,
        actions=_HangingExecutor(outbox, args.hang_execute) if args.hang_execute else None,
        hang_remember_after=args.hang_remember_after,
    )

    async def go():
        async with app.Session(opts) as session:
            if args.resume:
                outs = await session.resume(args.resume)
                return outs[-1] if outs else None
            return await session.run_target(_target(args), run_id=args.run_id)

    try:
        out = asyncio.run(go())
    except lease.LeaseLost as exc:
        print(str(exc), file=sys.stderr)
        return 3
    if out is None:
        print("nothing to resume")
        return 0
    print(json.dumps({"run_id": out.run_id, "graph_id": out.graph_id, "status": out.status,
                      "digest": out.digest, "nodes": out.nodes}, indent=2))
    return EXIT.get(out.status, 1)


if __name__ == "__main__":
    raise SystemExit(main())
