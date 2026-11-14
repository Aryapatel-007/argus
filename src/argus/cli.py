"""`argus` command. Subcommands are added as the phases land.

    argus migrate
    argus run-legacy --target <id>      (Sprint 1 linear loop)
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="argus")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("migrate", help="apply pending migrations to data/argus.db")
    legacy = sub.add_parser("run-legacy", help="run the Sprint 1 linear loop")
    legacy.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if args.command == "migrate":
        from argus.storage import migrate

        migrate.main()
        return 0
    if args.command == "run-legacy":
        from argus.orchestrator import runner

        return runner.main(args.rest)
    return 2


if __name__ == "__main__":
    sys.exit(main())
