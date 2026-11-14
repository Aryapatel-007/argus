# BUILD_STATE

Resume file for the ARGUS_BUILD.md build. Read this first, then CLAUDE.md,
then ARGUS_BUILD.md.

## Where we are

- **Phase:** E (graph engine), step 1-3 written, not yet committed
- **Last completed:** Phase D, testing and audit skills
- **Test count:** 270 at the E2 commit (230 at Phase D) (179 at baseline). Must never drop.
- **Blocked:** no

## Phase log

### Phase A — done (2026-11-14), commit e90c596

- Installed and pinned: playwright 1.65.2 (+ Chromium headless shell),
  apscheduler 3.11.3, fastapi 0.162.7, jinja2 3.1.6, pytest 9.2.0,
  pytest-asyncio 1.5.0, hypothesis 6.170.15. Already present and pinned:
  mcp 2.0.0, sqlite-vec 0.1.9, ollama 0.6.2, httpx 0.28.1, uvicorn 0.52.4.
- Rejected after reading the wheels: browser-use (D-002), deepeval (D-004),
  streamlit (D-003). Browser server will be raw Playwright.
- Versions, models, disk: `notes/environment.md`.
- ARGUS_BUILD.md saved into the repo so later sessions can re-read it.

### Phase B — done (2026-11-14), commit 3a7d52d

- 10 agents in `.claude/agents/` (Opus; effort medium/high/xhigh). Usage
  policy D-007. Project agents load only at session start: in the session
  that created them, run their prompts through general-purpose with the
  agent's file as instructions and the same effort.

### Phase C — done (2026-11-14)

- Audit: `notes/audit.md` (1 critical, 3 high, 8 medium, 15 low).
- Fixed C1, H1, H2, H3 plus M1, M4, M5, L2, L3, L14. Reviewer pass found 2
  high + 5 medium problems in those fixes; all fixed. Fix table at the end of
  notes/audit.md. Decisions D-008 to D-011.
- New: migrations 003 (run leases, unique checkpoint index) and 004 (remember
  receipts); `storage/lease.py`; `pyproject.toml` + editable install
  (`pip install -e .`); `argus` CLI stub.
- Writer think probe written up: `notes/writer-probe-results.md` (closed:
  stay on think=False).
- Tests: 211 at the Phase C commit.

### Phase D — done (2026-11-14)

- Installed Trail of Bits `property-based-testing` (commit 149d7b21, effort
  raised low->medium). Rejected Cloudflare security-audit-skill and 4 others.
  Details: `notes/skills.md`.
- Wrote project skills: argus-security-audit (+ `scripts/security_scan.py`),
  argus-agentic-testing, argus-mcp-testing, argus-mutation-check
  (+ `scripts/mutation_check.py`).
- Security scan: 0 blocking. Mutation check: all mutants killed (it caught
  one vacuous test: `atomic()` could not tell BEGIN from BEGIN IMMEDIATE).
- Tests: 230 at the Phase D commit.

## Phase E progress

Spec: `notes/specs/graph-engine.md`. Package `src/argus/graph/`.

| Step | Status |
|---|---|
| 1 node type registry | done (`graph/nodes.py`) |
| 2 graph model + migration | done: migrations 005_graph, 006_models, 007_llm_calls_graph; `graph/store.py` (only writer of graph state), `graph/readiness.py` (pure readiness, chains, retries); 17 tests incl. gate-readiness property + negative control |
| 3 validator + property tests | done: (`graph/validator.py`, `templates.py`, `tests/test_graph_validator.py`, 23 tests incl. 2 negative controls) |
| 4-12 | not started |

Model pool (decided, D-012 pending in log): qwen3.5:9b, phi4-mini,
llama3.1:8b, gemma4:e4b (profiler decides if it is usable), nomic-embed-text.
New pulls (gemma3:4b, qwen3.5:2b) FAIL on Ollama 0.24.0 ("download the latest
version"). Not upgrading the engine mid-build; question for Arya at the end.

## Next steps

1. Step 4: `config/models.yaml`, `graph/registry.py`,
   `scripts/profile_models.py`; run the profiler on the real pool.
4. Steps 5-12 per ARGUS_BUILD.md 6.10.

## Backlog (medium/low)

From notes/audit.md, not fixed: M2 (ctx mutated before commit), M3 (one
failed task fails the run), M6 (log_call failure kills the step), M7
(untyped memory-server errors), M8 (sync sqlite in async tools), L1, L4-L13,
L15; review L-4 (H1 negative control is query-level), L-7. Most of M2/M3/L4/
L7 are superseded by the graph engine.

## Questions for Arya (non-blocking)

- Upgrade Ollama 0.24.0 -> 0.40.x? Needed to pull newer small models
  (qwen3.5:2b, gemma3:4b). Would invalidate earlier measurements; the
  profiler would be re-run.
