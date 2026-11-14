# BUILD_STATE

Resume file for the ARGUS_BUILD.md build. Read this first, then CLAUDE.md,
then ARGUS_BUILD.md.

## Where we are

- **Phase:** G. Run 1 is done: **learning loses to the baseline**. A
  debugger fix is next.
- **Branch:** main. Phases A-F are merged; `wip/phase-e` and `wip/phase-f`
  are kept.
- **Test count:** 480 passing. It never drops.
- **Mutation check:** 27/27 killed. **Security scan:** 0 blocking
  (`notes/security-audit-phase-f.md`).
- **Blocked:** no. This becomes `BLOCKED: needs Arya` only if learning still
  loses after a real fix and a rerun (ARGUS_BUILD 8).

## Phase G run 1 (2026-11-15), see notes/routing-results.md

| | baseline | fixed | learning |
|---|---|---|---|
| Quality | 0.958 | 0.867 | 0.913 |
| Seconds per scenario | 46 | 87 | 92 |
| Model swaps | 0 | 112 | 112 |

- Gate violations were 0 in every mode.
- **The loss is in the structure scenarios:**
  - s14, s15 and s20 lost because work was routed to phi4-mini. It handled
    plan, extract and critique worse than qwen.
  - s13 lost because of a harness bug, now fixed (L2).
- **Root-cause hypothesis:** the reward is "passed validation", which is
  nearly always 1 (extract/phi4 α102 β5). The bandit therefore cannot see
  quality, and the cost-aware router prefers the cheaper model. Mixing
  models also costs about 4.7 swaps per scenario, because qwen3.5:9b
  co-resides with nothing.

## Next

1. **Debugger investigation** (ARGUS_BUILD 8). Use `debugger` at xhigh on
   the reward signal and the router.
   - Use only principled fixes, no tuning to the 24 scenarios. Candidates:
     - credit downstream outcomes back to the producing node (critic
       rejection, cross-model disagreement, an empty retry);
     - make swap cost per scenario count against switching;
     - require evidence before an arm displaces the profiler's
       quality-best model on a role.
   - Re-measure with `python scripts/phase_g.py`, run detached, with no
     other Ollama work at the same time. Unload the judge model afterwards
     (`ollama stop gemma4:e4b`), because it holds about 7 GB of RAM.
2. If learning still loses, write `BLOCKED: needs Arya` here with the
   numbers.
3. Check the definition of done (ARGUS_BUILD 9) item by item. Update
   README, CLAUDE.md and this file.

## Second delta review (of a602fb2): fixed and committed

- **H1, M3, M4:** `still_stated` (D-029).
- **H2, M1, M2, L5, L6:** browser (D-030).
- **L1, L2:** harness.
- **L3:** scheduler.
- **Tests:** `tests/test_delta2_review.py`.

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
