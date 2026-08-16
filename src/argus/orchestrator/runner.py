"""CLI for the orchestrator loop.

    python -m argus.orchestrator.runner --target test_target
    python -m argus.orchestrator.runner --resume <run_id>

Exits non-zero when the run ends FAILED, so a caller can detect failure without
parsing stdout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # direct-script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from argus.orchestrator import loop  # noqa: E402
from argus.orchestrator.state import State  # noqa: E402
from argus.storage import db  # noqa: E402
from argus.storage.migrate import apply_migrations  # noqa: E402


def _report(ctx) -> int:
    print(f"run_id:     {ctx.run_id}")
    print(f"target:     {ctx.target_id}")
    print(f"final state:{ctx.state.value:>8}")
    print(f"steps:      {ctx.step_index}")
    print(f"facts:      {len(ctx.facts_staged)} staged")
    if ctx.reflections:
        print("reflections:")
        for reflection in ctx.reflections:
            print(f"  - {reflection}")
    return 1 if ctx.state is State.FAILED else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="argus.orchestrator.runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target", help="start a fresh run against this target id")
    group.add_argument("--resume", metavar="RUN_ID", help="resume from the last checkpoint")
    parser.add_argument("--db", default=None, help="database path (default: data/argus.db)")
    args = parser.parse_args(argv)

    db_path = Path(args.db) if args.db else db.DB_PATH
    apply_migrations(db_path)

    with db.connect(db_path) as conn:
        if args.target:
            ctx = loop.new_run(args.target)
        else:
            ctx = loop.load_latest_checkpoint(args.resume, conn=conn)
            if ctx is None:
                print(f"no checkpoints found for run_id {args.resume!r}", file=sys.stderr)
                return 2
            if ctx.state in loop.TERMINAL_STATES:
                print(f"run {args.resume} already finished ({ctx.state.value})")
                return _report(ctx)
            print(f"resuming {args.resume} from step {ctx.step_index} ({ctx.state.value})")

        loop.run_to_completion(ctx, conn=conn)

    return _report(ctx)


if __name__ == "__main__":
    raise SystemExit(main())
