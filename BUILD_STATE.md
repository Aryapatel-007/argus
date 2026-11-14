# BUILD_STATE

Resume file for the ARGUS_BUILD.md build. Read this first, then CLAUDE.md,
then ARGUS_BUILD.md.

## Where we are

- **Phase:** G (prove the complexity earns its place) — benchmark RUNNING
- **Branch:** main. Phases A-F merged (`wip/phase-e`, `wip/phase-f` kept).
- **Test count:** 409 passing. Never drops.
- **Mutation check:** 22/22 killed. **Security scan:** 0 blocking
  (`notes/security-audit-phase-f.md`).
- **Blocked:** no

## In flight (check these first when resuming)

1. `scripts/phase_g.py` launched detached (Start-Process) at ~17:37 on
   2026-11-14, logs `data/phase-g.log` / `data/phase-g.err.log`. Order:
   judge calibration -> baseline -> fixed -> learning warm-up -> learning.
   Outputs: `notes/eval/*.json`, `notes/eval/judge-calibration.json`,
   `notes/eval/phase-g-table.md`. Eval DBs in `data/eval/`.
   If it died: rerun `python scripts/phase_g.py` (each mode starts fresh).
   Do NOT run the test suite, the mutation check or other Ollama work while
   it runs: wall-clock time is one of the measured metrics.
2. A delta review of the Phase F fixes + eval harness (commit a26c456) was
   running read-only. If its findings are not reflected in the log below,
   re-run the `reviewer` agent on `git show a26c456` after Phase G finishes.

## Then

- Write `notes/routing-results.md`: the table from
  `notes/eval/phase-g-table.md` plus an honest reading (where the graph
  engine wins or loses: quality, wall clock, swaps, timeouts, gate
  violations = 0 required in all three). If the learning engine does not
  beat the baseline after real effort: `BLOCKED: needs Arya` (ARGUS_BUILD 8).
- Check the definition of done (ARGUS_BUILD 9) item by item and update
  README/CLAUDE.md/this file.

## Phase log

- **A** (e90c596) deps pinned, Browser Use/deepeval/Streamlit rejected.
- **B** (3a7d52d) 10 Opus agents (D-007).
- **C** (30aade1) audit, C1/H1-H3 fixed, migrations 003-004.
- **D** (687ce95) skills: Trail of Bits PBT + 4 project skills, scanner,
  mutation check.
- **E** (merged d1e7ace) graph engine, router, strategies, real-kill crash
  tests, profiler. Three reviewer passes. D-012..D-024.
- **F** (merged e1b13a7) memory + browser MCP servers spawned per run (kill
  tests verify no orphans across the whole process tree), watch, approvals
  CLI + dashboard, observability, APScheduler with hard-kill recovery test,
  eval harness (24 scenarios, judge + calibration), README, demo. Reviewer
  pass fixed 1 high + 4 medium. D-025, D-026.

## Measurements worth knowing

- `notes/model-profiles.md`, `notes/phase-e-measurements.md`: phi4-mini ~81
  tok/s fully on GPU; qwen3.5:9b ~25 tok/s, 71% on GPU; llama3.1 ~51 tok/s;
  gemma4:e4b 32% on GPU. Co-residency is load-order-dependent; with Chromium
  running only llama3.1+nomic and phi4-mini+nomic are co-resident both ways.
- First real-model smoke (s02, baseline): quality 0.75, 72s per two-run
  scenario.

## Backlog (medium/low)

- Audit leftovers: M7 (typed memory-server errors), M8 (sync sqlite in async
  tools), L1, L4-L13, L15.
- Browser: hostnames that resolve to private IPs are not filtered (IP
  literals and localhost are). Accepted while targets come from the owner.
- Profiler think/load calls write no llm_calls rows; `backup_if_pending`
  name collision within one second.
- Fixtures in `eval/node_fixtures.py` saturate (most models score 1.0); the
  24 scenarios are the harder set.

## Questions for Arya (non-blocking)

- Upgrade Ollama 0.24.0 -> 0.40.x? Needed to pull newer small models
  (qwen3.5:2b, gemma3:4b). Would invalidate earlier measurements; the
  profiler would be re-run.
