# BUILD_STATE

Resume file for the ARGUS_BUILD.md build. Read this first, then CLAUDE.md,
then ARGUS_BUILD.md.

## Where we are

- **Phase:** G — STOPPED before a valid benchmark; fix delta-review findings first
- **Branch:** main. Phases A-F merged (`wip/phase-e`, `wip/phase-f` kept).
- **Test count:** 409 passing. Never drops.
- **Mutation check:** 22/22 killed. **Security scan:** 0 blocking
  (`notes/security-audit-phase-f.md`).
- **Blocked:** no

## STOPPED (usage limit, 2026-11-14 ~17:55). Resume here.

Phase G run was STOPPED on purpose after 13 baseline scenarios (all q=1.0
except s13 0.33, ~40-67s each, 0 swaps, 0 gate violations; see
data/phase-g.log). Its numbers must NOT be used: a delta review of commit
a26c456 found flaws that bias the benchmark. Fix these FIRST, with tests,
then rerun `python scripts/phase_g.py` (detached; no other Ollama work
while it runs). Judge calibration from that run: gemma4:e4b 0.83, qwen3.5:9b
0.79, phi4-mini 0.29, llama3.1 0.29 (check judge-calibration.json: 0.29 may
be invalid-JSON counted as disagreement).

Delta review findings (all open):
- H-B `llm_nodes.still_stated` unanchored substring: "IELTS overall 6" is
  "stated" on a page saying 6.5; also 12 vs 120, 45,000 vs 45,000,000, list
  extended after the old text; a stored "." matches everything. Anchor on
  word/number boundaries + sentence end; empty canon = not stated; tests.
- H-A browser.py: `context.route` sees only the FIRST url of a redirect
  chain, so HTTP 30x hops (top-level goto and subresources) reach internal
  hosts; robots fetch via context.request follows redirects unrouted. Fix:
  route.fetch(max_redirects=0) + re-check Location, max_redirects=0 on
  robots; add HTTP-302 tests (subresource and top level) with `other.hits`.
- M-A s08 needs `silent=True, kinds={"fees": "unchanged"}`; consider a
  symmetric equivalence check in still_stated (same numbers + mutual high
  overlap) so a re-worded extraction is not "modified".
- M-B scheduler.reflect (and start_run failing after its first commit)
  strands the lease like M1 did; share the release helper; test.
- M-C recover cap: in-memory, permanent, silent. Use backoff + a visible
  marker (runs.status 'parked' + dashboard row).
- M-D harness: judge loads a different model between scenarios, biasing the
  next scenario's routing and wall time. Judge AFTER all scenarios of a mode.
- M-E harness: score "all expected runs completed with acceptable graph
  status" (except s24); today a failed last run can pass vacuously.
- M-F harness gate metric: count any approval status != pending (incl.
  failed) or resolved_at set; scan approvals.proposed_action for forbidden
  tokens (s12 exists for that).
- M-G browser: http->https upgrade refused (port in origin key); align with
  handlers.same_site.
- M-H browser `_internal`: trailing-dot localhost, 100.64/10 (use
  `not ip.is_global`); DNS-to-private names remain a documented residual.
- Lows: old Chromium/driver not closed on relaunch; IDN punycode vs unicode
  host; robots network error should mean disallow; busy-loop test assertion
  weak; judge gets pipeline diff items as "ground truth" (state this in the
  Phase G write-up); procedural memory contributes nothing to the measured
  learning pass (state it).

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
