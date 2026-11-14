# BUILD_STATE

Resume file for the ARGUS_BUILD.md build. Read this first, then CLAUDE.md,
then ARGUS_BUILD.md.

## Where we are

- **Phase:** E done and merged to main. F in progress (steps 1-2 done).
- **Branch:** main (`wip/phase-e` merged; kept, not deleted).
- **Test count:** 368 passing. Never drops.
- **Mutation check:** 20/20 killed. **Security scan:** 0 blocking.
- **Blocked:** no

## Phase log

- **A** (e90c596) deps pinned, Browser Use/deepeval/Streamlit rejected
  (telemetry). `notes/environment.md`.
- **B** (3a7d52d) 10 Opus agents in `.claude/agents/` (D-007).
- **C** (30aade1) `notes/audit.md`: C1 gate-after-tool, H1-H3 fixed, plus
  reviewer findings on the fixes. Migrations 003 (leases), 004 (receipts).
- **D** (687ce95) Trail of Bits property-based-testing installed; 4 project
  skills; `scripts/security_scan.py`, `scripts/mutation_check.py`.
- **E** steps 1-3 (ab58aeb), step 2 (2cc6de3) on main; steps 4-12 on
  wip/phase-e:
  - config: `config/models.yaml` (allowlist, named `baseline` and `embed`,
    no tiers, D-021), `config/routing.yaml` (all router/scheduler knobs).
  - `graph/llm.py` async backend (think=False top-level, 120s wall cap,
    every call logged), `llm_nodes.py` + `prompts/` (all page-derived text
    fenced and escaped), `registry.py`, `router.py` (score breakdown stored,
    Thompson sampling with bounded exploration, swap-aware, D-017/D-018),
    `bandit.py` (rewards in the node's own transaction), `health.py`
    (half-open breaker), `strategies.py` (single, reflexion_retry, cascade,
    cross_model, self_consistency, low-confidence second opinion; samples
    stored and reused), `handlers.py` (all node types; executor claims the
    approval first, D-023), `engine.py` (two lanes, swap-aware batching,
    retries as nodes, triggers -> replan, gate resolution, budgets,
    lease per graph), `procedural.py` (template outcomes, learned hints shown
    to the planner), `fakes.py`, `runner.py` (CLI).
  - `scripts/profile_models.py` ran on the real pool: `notes/model-profiles.md`,
    findings in `notes/phase-e-measurements.md` (co-residency is
    load-order-dependent; think="low" still thinks).
  - Tests: engine end to end, router/bandit/health, expansion + procedural,
    review fixes, and REAL-KILL crash tests (`tests/test_graph_crash.py`):
    7 in-call kill points + 2 execute windows + live-lease refusal.
  - Two reviewer passes. First: 2 high + 8 medium, all fixed
    (`tests/test_graph_review_fixes.py`). Delta review of those fixes: 1 high
    (a malformed port crashed the engine on every resume) + 3 medium, all
    fixed (`tests/test_graph_delta_fixes.py`). Decisions D-016 to D-024.
- **F** step 1: `graph/mcp_clients.py` (McpMemory/McpBrowser over stdio,
  RunMemory owns the per-run spawn; runner defaults to `--memory mcp`).
  Step 2: every graph crash test now runs the memory server as a real MCP
  subprocess, including the kill right after `remember` committed, and
  asserts by process listing that the server dies with its killed parent
  (Windows Job Object verified, not assumed).

## Next steps

1. Phase F (ARGUS_BUILD.md section 7), remaining:
   3. Browser MCP server (`servers/browser.py`, raw Playwright, D-002):
      `open_page` returns text + links + confidence, degrades instead of
      raising, per-domain throttle, real wait conditions, closes contexts.
      Then re-measure co-residency with Chromium running.
   4. Watch logic over config/targets.yaml (one graph per target per run).
   5. Approval CLI (`argus approvals list/approve/reject`) + dashboard.
   6. Dashboard (FastAPI + Jinja2, 127.0.0.1, CSRF on POST): runs, trace
      tree, routing explanations, approvals, model health.
   7. Eval harness: 20-30 scenarios on local fixture pages, judge != judged
      model, calibration against a hand-labelled set.
   8. APScheduler + hard-kill test + reflect (consolidation) schedule.
   9. One-command start, README, demo script. Security scan again.
2. Phase G: baseline vs fixed vs learning on identical scenarios ->
   `notes/routing-results.md`.

## Backlog (medium/low)

- Audit leftovers: M2, M3 (superseded by the graph engine), M6 (legacy
  log_call), M7 (typed memory-server errors), M8 (sync sqlite in async
  tools), L1, L4-L13, L15.
- Review lows not yet done: profiler think/load calls write no llm_calls rows
  and use only httpx per-phase timeouts; `backup_if_pending` name collision
  within one second; bandit/health/log_call writes are not lease-guarded
  (observability/learning rows, deliberate).
- Fixtures saturate (most node types score 1.0 for every model); the eval
  harness must add harder cases so Phase G can separate models.

## Questions for Arya (non-blocking)

- Upgrade Ollama 0.24.0 -> 0.40.x? Needed to pull newer small models
  (qwen3.5:2b, gemma3:4b). Would invalidate earlier measurements; the
  profiler would be re-run.
