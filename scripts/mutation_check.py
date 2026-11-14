"""Mutation-style check for Argus's critical guarantees.

    python scripts/mutation_check.py            # all mutants
    python scripts/mutation_check.py gate lease # mutants whose name contains these

Each mutant plants ONE known bug in a copy of the repo (never the working
tree) and runs only the tests meant to guard that guarantee. A mutant that
survives (tests still pass) means those tests are vacuous for that bug.

Why hand-written mutants instead of a mutation tool: the Trail of Bits
mutation skill drives mewt, which has no Python support and does not run on
Windows (notes/skills.md). A short list of the bugs that would actually hurt
Argus is more useful than thousands of random operator swaps.

Exit 1 if any mutant survives, or if a mutant's anchor text is missing
(the guarded code moved and the mutant must be updated).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

REPO = Path(__file__).resolve().parents[1]
COPY = ["src", "tests", "scripts", "pytest.ini", "config"]


@dataclass(frozen=True)
class Mutant:
    name: str
    path: str
    old: str
    new: str
    tests: Tuple[str, ...]


MUTANTS: List[Mutant] = [
    Mutant("validator-gate-check-off", "src/argus/graph/validator.py",
           "    for n in gate_bypasses(graph):",
           "    for n in []:",
           ("tests/test_graph_validator.py",)),
    Mutant("readiness-ignores-unapproved-gates", "src/argus/graph/readiness.py",
           'if not gates or any(nodes[g].status != "done" for g in gates):',
           "if not gates:",
           ("tests/test_graph_store.py",)),
    Mutant("decision-not-frozen-at-start", "src/argus/graph/store.py",
           '"UPDATE routing_decisions SET frozen = 1 WHERE node_id = ? AND superseded = 0"',
           '"UPDATE routing_decisions SET frozen = 0 WHERE node_id = ? AND superseded = 0"',
           ("tests/test_graph_store.py",)),
    Mutant("loop-gate-after-tool", "src/argus/orchestrator/loop.py",
           "        return State.GATE, ops\n    return State.ACT, ops",
           "        return State.ACT, ops\n    return State.ACT, ops",
           ("tests/test_audit_fixes.py",)),
    Mutant("tool-side-effect-flag-ignored", "src/argus/orchestrator/tools.py",
           "    return bool(planner_flag) or tool_name in SIDE_EFFECT_TOOLS",
           "    return bool(planner_flag)",
           ("tests/test_audit_fixes.py",)),
    Mutant("lease-check-off", "src/argus/storage/lease.py",
           "    if cursor.rowcount != 1:\n        raise LeaseLost",
           "    if False:\n        raise LeaseLost",
           ("tests/test_audit_fixes.py",)),
    Mutant("vectorless-fact", "src/argus/servers/memory.py",
           "            if needs_new_row and vector_blob is None:\n                raise _NeedsEmbedding()",
           "            if False:\n                raise _NeedsEmbedding()",
           ("tests/test_audit_fixes.py",)),
    Mutant("receipt-not-written", "src/argus/servers/memory.py",
           "            if idempotency_key:\n                conn.execute(\n                    \"INSERT INTO remember_receipts",
           "            if False:\n                conn.execute(\n                    \"INSERT INTO remember_receipts",
           ("tests/test_audit_fixes.py",)),
    Mutant("deferred-begin", "src/argus/storage/db.py",
           'conn.execute("BEGIN IMMEDIATE")',
           'conn.execute("BEGIN")',
           ("tests/test_storage.py",)),
    Mutant("thinking-on", "src/argus/orchestrator/llm.py",
           "THINK = False",
           "THINK = True",
           ("tests/test_llm.py",)),
    Mutant("checkpoint-outside-transaction", "src/argus/orchestrator/loop.py",
           "        with db.atomic(active):\n            for sql, params in ops:",
           "        if True:\n            for sql, params in ops:",
           ("tests/test_storage.py", "tests/test_loop.py", "tests/test_audit_fixes.py")),
]


def _copy_repo(dest: Path) -> None:
    for name in COPY:
        src = REPO / name
        if src.is_dir():
            shutil.copytree(src, dest / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        elif src.exists():
            shutil.copy2(src, dest / name)


def run_mutant(m: Mutant) -> str:
    with tempfile.TemporaryDirectory(prefix="argus-mut-") as tmp:
        root = Path(tmp)
        _copy_repo(root)
        target = root / m.path
        if not target.exists():
            return "NOT-PRESENT"
        text = target.read_text(encoding="utf-8")
        if m.old not in text:
            return "ANCHOR-MISSING"
        target.write_text(text.replace(m.old, m.new, 1), encoding="utf-8")
        env = dict(os.environ)
        # The copy's src must win over the editable install of the real repo.
        env["PYTHONPATH"] = str(root / "src")
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
             "-m", "not ollama", *m.tests],
            cwd=root, env=env, capture_output=True, text=True, timeout=900,
        )
        return "KILLED" if proc.returncode != 0 else "SURVIVED"


def main(argv: List[str] | None = None) -> int:
    filters = sys.argv[1:] if argv is None else argv
    chosen = [m for m in MUTANTS if not filters or any(f in m.name for f in filters)]
    bad = 0
    for m in chosen:
        status = run_mutant(m)
        bad += status not in ("KILLED", "NOT-PRESENT")
        print(f"{status:15} {m.name}  ({m.path})", flush=True)
    print(f"\n{len(chosen) - bad}/{len(chosen)} mutants killed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
