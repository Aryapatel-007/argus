"""Phase G: does the graph engine earn its complexity? (ARGUS_BUILD 8)

    python scripts/phase_g.py                 # everything (1-3 hours on this laptop)
    python scripts/phase_g.py --scenarios s02_fee_up s10_injection   # a quick slice

1. Calibrate the judge: every pool model on the hand-labelled set.
2. baseline  everything on the baseline model (qwen3.5:9b), single strategy,
             no learning: the Sprint 1 design, run on the graph engine.
3. fixed     graph engine with profiler priors, learning off.
4. learning  graph engine with Thompson sampling: one warm-up pass over the
             scenarios (rewards accumulate), then the measured pass.
Each mode starts from a fresh database seeded with the SAME measured
profiles and no learned arms. Identical scenarios, identical pages.

Writes notes/eval/<mode>.json, notes/eval/judge-calibration.json and the
numbers table in notes/routing-results.md. The written reading of the
numbers is added by hand afterwards: the script reports, it does not judge.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from argus import config  # noqa: E402
from argus.eval import harness, judge  # noqa: E402
from argus.eval.scenarios import SCENARIOS, by_id  # noqa: E402
from argus.graph.llm import LogContext, OllamaBackend  # noqa: E402
from argus.storage import db  # noqa: E402

OUT = REPO / "notes" / "eval"
EVAL_DB = db.DB_PATH.parent / "eval"


def log(msg):
    print(f"[phase-g {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fresh(name: str) -> Path:
    path = EVAL_DB / f"{name}.db"
    for suffix in ("", "-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)
    harness.copy_profiles(db.DB_PATH, path)
    return path


async def calibrate(models):
    path = fresh("judge")
    with db.connect(path) as conn:
        from argus.storage import db as dbm
        with dbm.atomic(conn):
            conn.execute("INSERT INTO runs (run_id, target_id, status) VALUES ('judge-cal', 'judge', 'running')")
        return await judge.calibrate(OllamaBackend(), models, LogContext(run_id="judge-cal", role="judge", conn=conn))


async def suite(mode, path, scenarios, judges, tag=""):
    log(f"--- {mode}{tag}")
    return await harness.run_suite(db_path=path, mode=mode, scenarios=scenarios, judge_models=judges,
                                   tag=tag, progress=lambda m: log(m))


def table(reports):
    keys = [("quality_mean", "Quality (mean of checks)"), ("gate_violations", "Approval-gate violations"),
            ("wall_s_total", "Wall clock, all scenarios (s)"), ("wall_s_mean", "Wall clock per scenario (s)"),
            ("swaps_total", "Model swaps"), ("llm_calls_total", "Model calls"), ("timeouts_total", "Timeouts"),
            ("retries_total", "Retried nodes"), ("critic_self_judged", "Critic judged its own model"),
            ("deterministic_faithfulness", "Digest numbers grounded"),
            ("judge_faithful_rate", "Judge: faithful"), ("judge_complete_rate", "Judge: complete")]
    modes = list(reports)
    lines = ["| metric | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for k, label in keys:
        lines.append(f"| {label} | " + " | ".join(str(reports[m]["summary"].get(k)) for m in modes) + " |")
    cats = sorted({c for m in modes for c in reports[m]["summary"]["quality_by_category"]})
    for c in cats:
        lines.append(f"| Quality: {c} | " + " | ".join(str(reports[m]["summary"]["quality_by_category"].get(c))
                                                     for m in modes) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*")
    ap.add_argument("--skip-calibration", action="store_true")
    args = ap.parse_args(argv)
    scenarios = [by_id(s) for s in args.scenarios] if args.scenarios else list(SCENARIOS)
    models = config.allowed_models()
    OUT.mkdir(parents=True, exist_ok=True)
    EVAL_DB.mkdir(parents=True, exist_ok=True)

    judges = models
    if not args.skip_calibration:
        log("calibrating the judge")
        cal = asyncio.run(calibrate(models))
        (OUT / "judge-calibration.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
        judges = sorted(models, key=lambda m: -cal[m]["agreement"])
        log(f"judge order by agreement: {[(m, round(cal[m]['agreement'], 2)) for m in judges]}")

    reports = {}
    reports["baseline"] = asyncio.run(suite("baseline", fresh("baseline"), scenarios, judges))
    reports["fixed"] = asyncio.run(suite("fixed", fresh("fixed"), scenarios, judges))
    learn_db = fresh("learning")
    reports["learning-warmup"] = asyncio.run(suite("learning", learn_db, scenarios, judges, tag="-warmup"))
    reports["learning"] = asyncio.run(suite("learning", learn_db, scenarios, judges))
    for name, rep in reports.items():
        (OUT / f"{name}.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")

    numbers = table({k: v for k, v in reports.items()})
    stamp = time.strftime("%Y-%m-%d %H:%M")
    (OUT / "phase-g-table.md").write_text(f"Generated {stamp}, {len(scenarios)} scenarios.\n\n{numbers}\n",
                                          encoding="utf-8")
    log("table written to notes/eval/phase-g-table.md")
    print(numbers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
