# Argus audit (Phase C)

Auditor pass, 2026-11-14, read-only (auditor role, `.claude/agents/auditor.md`,
run through a general-purpose Opus agent because project agents load only at
session start). Baseline: `python -m pytest -q` -> 179 passed. Ollama was up,
so real-embed tests ran. "Demonstrated" means reproduced against a temp
database. `data/argus.db` was never opened. Not audited: `src/argus/graph/`,
`tests/test_graph_validator.py`, `notes/specs/` (written during the audit).

Fix status is tracked at the bottom of this file and in BUILD_STATE.md.

---

## 1. Map

- **config.py** — loads `config/targets.yaml` (lru_cache). `model_for_role`
  maps role -> tier key (`chat`, `fast`, `embed`) -> tag; optional `roles:`
  override. `ollama_options()` gives host and num_ctx.
- **ollama_client.py** — cached sync `Client` per (host, timeout); cached
  `AsyncClient` per (loop, host, timeout) via WeakKeyDictionary;
  `TIMEOUT_SECONDS=120`; `log_call`, the single INSERT path into `llm_calls`,
  autocommit, outside any transaction.
- **embeddings.py** — async embed, checks count and 768 width, writes one
  `llm_calls` row (role `embedder`, think_mode `n/a`) before raising, named
  errors. `serialize()` for sqlite-vec. Never imports the orchestrator.
- **storage/db.py** — `connect()`: WAL, busy_timeout 5000, FKs, synchronous
  NORMAL, `isolation_level=None`, sqlite-vec. `atomic()`: BEGIN IMMEDIATE,
  COMMIT or ROLLBACK on any BaseException.
- **storage/migrate.py + migrations/** — files in name order, each in its own
  `atomic()` unless the first line is `-- migrate: no-transaction` (001c).
  001 runs/tasks/checkpoints/llm_calls; 001b adds and 001c removes
  `tasks.idempotency_key`; 001c rebuilds tasks and llm_calls with CHECKs; 002
  adds facts (versioned, partial unique current index), facts_vec (768),
  approvals (UNIQUE idempotency_key), rebuilds llm_calls for `embedder`.
- **orchestrator/state.py** — `State` enum (INIT..FAILED) and the separate
  task lifecycle strings.
- **orchestrator/context.py** — `RunContext`, the whole resume unit, JSON in
  `checkpoints.payload` every step.
- **orchestrator/loop.py** — linear state machine. Handler mutates ctx and
  returns (next_state, ops); `step()` runs the handler (model calls happen
  here) then writes ops + checkpoint + terminal runs update in one
  `atomic()`. Guards MAX_TASKS_PER_RUN, MAX_STEPS, MAX_ATTEMPTS,
  MAX_PARSE_RETRIES. Resume = newest checkpoint by step_index.
- **orchestrator/llm.py** — sync chat, `think=False` top level, `format=`
  schema, temperature 0, keep_alive 30m, one schema retry, one log row per
  attempt, increments `ctx.parse_retries` on final failure.
- **orchestrator/fake_llm.py, tools.py** — `fake_observe` really runs stub
  tools; three stub fetch tools (ok, fails, empty).
- **orchestrator/runner.py** — CLI `--target/--resume/--db`, migrates on every
  start, exit 1 on FAILED.
- **servers/memory.py** — stdio MCP server (v2 `MCPServer`), `--db-path`,
  `--run-id`; startup checks migration 002 and the run row; three async tools,
  fresh connection each: `remember_fact` (hash, embed outside tx,
  double-checked read+write in `atomic()`), `recall_facts` (SQL prefilter,
  embed, KNN over all eligible, relevance/recency blend),
  `list_changes_since` (created_at feed, no embedding).
- **servers/scoring.py** — relevance `1 - L2^2/4`, recency 30-day half-life,
  0.7/0.3 blend, importance deliberately absent.
- **Tests** — storage (migrations, CHECKs, atomic + negative control,
  BEGIN IMMEDIATE timing, two writer threads), llm (params, logging, retry,
  prompt/schema drift), loop (full run, contiguity, resume, guards, gate
  stub), embeddings (real width, determinism, failures, cache), memory server
  (all tools via in-memory MCP Client, forced race, startup checks, stdout
  discipline).
- **Scripts** — `crash_test.py` real OS-kill harness (manual), probes, smoke.

```
runner -> migrate -> loop.step -> handlers -> llm.call_ollama -> ollama_client
                                \-> fake_observe -> tools.run_tool (stubs)
                                \-> db.atomic(ops + checkpoint)
memory server (separate process, spawned by nothing yet) -> embeddings -> ollama_client
                                                         -> db.connect/atomic -> facts
```

The orchestrator and memory server share only the SQLite file and
`ollama_client`. No MCP link between them.

---

## 2. Claims without proof

| # | Claim | What tests it | Gap |
|---|---|---|---|
| 1 | crash mid-step leaves DB as before (`db.py:10-12`, `loop.py:366-372`) | raise-inside-with tests | No OS-kill test in the suite; `crash_test.py` is manual; between-steps window INCONCLUSIVE |
| 2 | PLAN->PLAN on parse failure, PLAN->FAILED at MAX_PARSE_RETRIES | only `call_ollama` counter | Nothing drives the loop through two bad plans |
| 3 | GATE->FAILED "blocked pending approval" | run FAILED + error text | Never checks the tool did not run. **It ran 3 times** (C1) |
| 4 | `roles:` override and direct-tag form (`config.py`) | nothing | untested |
| 5 | model tags from config, never source | circular tests comparing against `model_for_role()` | a hardcoded tag would still pass |
| 6 | double read protects the race (`memory.py:388-398`) | in-process race where every racer embeds | stale-advisory path untested and broken (H1) |
| 7 | fact/facts_vec correspondence (`002:106-108`) | nothing | broken by H1 |
| 8 | `PRAGMA foreign_key_check` in 002 | FK check after migrate | result discarded by migrate.py; can never fail |
| 9 | two processes x 200 writes (CLAUDE.md) | two **threads** x 100 | process proof only in a probe script |
| 10 | `run_tool` never raises | ToolError only | other exceptions escape |
| 11 | exactly one llm_calls row per call | `conn=None` writes none | async 120s timeout never asserted |
| 12 | run row exists by construction when spawned (`memory.py:225-230`) | nothing | nothing spawns the server yet |
| 13 | MAX_STEPS tracks its inputs | nothing | STEPS_PER_TASK is a literal |
| 14 | llm_calls row survives a rolled-back step | manual crash test | no pytest test |
| 15 | one transaction per migration | single process only | concurrent migrators fail (M5) |

Properly proven (do not re-check): think=False/timeout/format/no num_predict
(`test_llm.py:129-155`), timeout still logs, BEGIN IMMEDIATE with negative
control, user_version not used, partial unique index on facts, recall
prefilter before KNN, change feed created_at semantics, stdout discipline,
--run-id startup check.

---

## 3. Known open items

| Item | Status |
|---|---|
| Writer quality probe | **Closed** (`notes/writer-probe-results.md`, verified against raw JSON). Residual: soft over-linking on thin evidence; deterministic number/date grounding check proposed. Writer not wired in the legacy loop. |
| Single-shot planner | Open, structural. Graph `replan` replaces it. Planner never receives the target URL or extract fields; `targets.yaml` url/extract unused by code. |
| Orchestrator has no MCP client | Open. Memory server only ever ran under Inspector. |
| Crash-resume across subprocess boundary | Open; design gap H3. |
| Windows orphan processes | Open, unverified. `crash_test.py` kills one PID. MCP v2 stdio client uses a Job Object; verify by process listing. |
| entity_key drift | Open. Exact match; drift creates a new `added` fact and the old stays current forever (nothing writes `removed`). |
| Co-residency with a browser | Open, unmeasured. |
| Importance scoring | Open by design; `test_memory_server.py:519-530` pins its absence and must change deliberately. |
| Between-steps crash window | INCONCLUSIVE (~3-6ms). |
| Retrieval quality | Unresolved (2 of 5 phrasings). |
| `removed` change kind | Nothing writes it; must set supersedes_id. |
| Approval flow | Not built; GATE is a stub. |
| Crashed runs | `runs.status` stays `running` forever (H2). |
| Ollama engine | 0.24 -> 0.40 deferred. |
| Telemetry | `opentelemetry-api` arrives with mcp; no SDK/exporter installed, so no-op. Never install `opentelemetry-sdk`. |

---

## 4. Flaws

### Checked and fine

- No transaction is held across any Ollama call (loop dispatches before
  `atomic()`; `remember_fact`'s advisory read is autocommit, BEGIN IMMEDIATE
  only after the await).
- `log_call` outside `atomic()` is an acceptable exception: a single
  autocommit INSERT takes the write lock up front, honours busy_timeout
  (measured), never upgrades a read lock, and stays outside so the row
  survives a rolled-back step. Only the written rule needs to name it (L8).
- Embed failure inside `remember_fact` degrades: `is_error=True` with the
  error text, nothing written, one llm_calls row `ok=0 timed_out=1`. Untyped
  (M7), untested.
- Chat path follows every convention. Memory server stdout discipline holds.

### Critical

**C1. The approval gate runs after the action, not before it.** OBSERVE runs
`tools.run_tool` (`loop.py:266-269`) before GATE (`loop.py:291-306`); external
is decided only by the planner's `externally_visible` flag. Demonstrated: a
counting `send_email` tool executed 3 times, then GATE failed the run. A
planner marking it false sails through. Latent today (stubs only), breaks hard
constraint 4 the moment a real external tool exists.

### High

**H1. Stale advisory read leaves the current fact with no embedding.**
`memory.py:423-434` skips the embed when the outside read says unchanged; the
in-transaction read can then take the modified path with `vector_blob=None`
and skip the facts_vec insert. Demonstrated: modified v3 with 0 facts_vec
rows; recall returns nothing for it, permanently. Needs a second process
writing the same entity between the reads (overlapping runs, orphaned server).

**H2. Nothing stops two processes driving the same run.** Checkpoint index
not unique, no lease, `runs.status` stays `running` after a crash. Both
processes write the same step_index; work runs twice; resume picks a branch
arbitrarily.

**H3. `remember_fact` is not idempotent across a crash.** Server commits on
its own connection before the orchestrator's checkpoint. On resume the
repeated call reports `unchanged` for a fact first seen this run, or writes a
phantom `modified` if the writer output differs. No caller idempotency key.

### Medium

- **M1.** `call_ollama` increments `ctx.parse_retries` for any role; critic
  failures consume the planner's retry budget.
- **M2.** `step()` mutates ctx before the commit; a caller that catches a
  failed commit and steps again skips an index and loses a popped task.
- **M3.** One failed task fails the whole run.
- **M4.** `python -m argus...` fails without PYTHONPATH (no pyproject, no
  editable install).
- **M5.** Concurrent migrators crash (applied set read once, outside tx).
- **M6.** A logging failure (lock timeout in `log_call`) kills the step
  after the model call was paid for.
- **M7.** Memory-server errors are untyped text; retryable vs permanent is
  indistinguishable without parsing strings.
- **M8.** Sync sqlite inside async tools: a 5s lock wait freezes the server's
  event loop.

### Low

- L1 `PRAGMA foreign_key_check` in 002 does nothing.
- L2 `memory.py:770` no-op expression.
- L3 `tools.py` catches only ToolError.
- L4 `STEPS_PER_TASK` literal.
- L5 config edge cases (null `roles:`, falsy override, misleading error for
  unlisted tag, `router` -> `fast` has no schema).
- L6 `atomic()` hides the original exception if ROLLBACK fails; a failed
  COMMIT leaves the connection mid-transaction.
- L7 None action ends the run leaving a pending task; `done=true` with tasks
  drops them.
- L8 docs drift (log_call exception unnamed in CLAUDE.md; runner `-m`
  docstring; importance "neutral constant" wording; inspector-session
  one-liners close the connection immediately — demonstrated).
- L9 unchanged path drops a new confidence/source_url.
- L10 recall `IN (...)` capped at 32766 vars; eligible set can go stale during
  the embed.
- L11 `list_changes_since` does not validate limit or kinds.
- L12 `conn=None` makes a call silently unlogged.
- L13 120s is a per-phase httpx timeout, not a wall-clock cap (fine while
  non-streaming).
- L14 `scripts/smoke_test.py` chat call has no think=False and no timeout.
- L15 memory server never closes async clients on shutdown.

---

## 5. What the graph engine will break

- **RunContext payload** assumes one linear cursor; `parse_retries` shared
  across roles; `reflections`/`facts_staged` copied in full every checkpoint
  (quadratic growth); `from_dict` raises on unknown keys. Keep it only for the
  ported linear template; graph state lives in `graph_nodes` rows.
- **`llm_calls.role` CHECK** admits six values; every node type would need a
  rebuild. No node/strategy/sample/routing columns; `task_id` FK to tasks.
- **Single-model config**: tier keys (`router: fast` is exactly the banned
  tier label); `model_for_role` is called inside `call_ollama`, so a router
  cannot pass its choice; models live in targets.yaml instead of an allowlist.
- **State enum** is linear; graph nodes need their own lifecycle. Do not
  overload `State` or `tasks.status`.
- **tasks table**: CHECK with four statuses and the attempts pairing; under
  retries-as-nodes `attempts` loses meaning. Leave `tasks` to the legacy loop.
- **approvals**: unused; `created_task_id` only; transitions not enforced.
  `execute_action` needs `UPDATE ... SET status='executed' WHERE key=? AND
  status='approved'` with rowcount check, in the same tx as the node
  transition. Gate must precede execution (C1).
- **log_call**: no node/strategy/sample/routing args. Keep one insert path,
  add optional kwargs.
- **MAX_* constants** are linear; replace with graph limits from config.
- **crash_test.py** assumes one run per DB, PLAN/CRITIQUE as LLM markers,
  contiguous step_index, ACT count = attempts, single-PID kill. A graph
  version needs node-level kill triggers and a process-tree check.
- **Sync call path**: the two-lane asyncio scheduler needs async model calls;
  `test_loop.MockOllama` routes on prompt prefixes.

---

## 6. Recommended fixes, ranked

| Sev | Fix | Test |
|---|---|---|
| Critical | Gate before the tool runs; externality = static per-tool flag OR planner flag; write approvals row and block | counting external tool flagged false: 0 executions, 1 pending approval; negative control |
| High | `remember_fact`: if a new row is needed and no vector, leave tx, embed, retry | simulated stale read; invariant: no current fact without a facts_vec row |
| High | UNIQUE(run_id, step_index) on checkpoints + run lease | two processes resume the same run: exactly one proceeds |
| High | `remember_fact` optional idempotency_key replaying the first result | same key twice: same result, change_kind still `added` |
| Medium | per-role failure counting | two critic timeouts then a planner parse failure: no FAIL |
| Medium | ctx mutation after commit | forced checkpoint failure, keep stepping: contiguous |
| Medium | pyproject.toml + `pip install -e .` | `python -m argus.servers.memory --help` without PYTHONPATH |
| Medium | migrate re-checks inside each file's tx | two concurrent `apply_migrations` both succeed |
| Medium | failed task terminates the task, not the run | superseded by the graph engine |
| Medium | typed memory-server error kinds | timeout -> retryable, empty -> invalid_input |
| Medium | `log_call` failure degrades | patched execute raises: step still commits |
| Medium | loop-level PLAN->PLAN->FAILED test | planner invalid 4x |
| Medium | sqlite off the event loop in the memory server | concurrent read returns under a held lock |
| Low | L1-L15 above | see each |

---

## Fix status

Phase C commit. Every critical and high fixed with a test and a negative
control (or a mutant in `scripts/mutation_check.py`). An independent reviewer
pass then found 2 high and 5 medium problems in the fixes themselves; all
fixed in the same commit.

| Finding | Fix | Proof |
|---|---|---|
| C1 gate after tool | PLAN -> GATE for external actions before ACT; GATE writes approvals row (`ON CONFLICT (idempotency_key) DO NOTHING`); `tools.SIDE_EFFECT_TOOLS` static flag OR planner flag; OBSERVE refuses external tools itself (review M-2); malformed tool names fail closed | `test_audit_fixes.py` C1 tests + negative control; mutants `loop-gate-after-tool`, `tool-side-effect-flag-ignored` |
| H1 vectorless fact | `_NeedsEmbedding` rollback, embed outside tx, re-decide | stale-read test; invariant query negative control; mutant `vectorless-fact` |
| H2 two drivers | migration 003: UNIQUE(run_id, step_index) + `run_leases`; heartbeat/ownership checked inside each step tx; runner acquires before loading the checkpoint (review M-3); same-host live pid never loses its lease to staleness (review H-2); access-denied reads as alive (review M-1) | lease tests incl. step ops rolled back; fork refused by index; negative control with index dropped; mutants `lease-check-off`, `checkpoint-outside-transaction` |
| H3 non-idempotent remember | migration 004 `remember_receipts` written in the fact's tx; receipts scoped to (target, entity), reuse raises (review M-5); server requires 004 | same-key replay test; no-key negative control; 002-without-004 refused; mutant `receipt-not-written` |
| M1 shared parse_retries | planner-only increment | `test_critic_failures_do_not_spend_the_planner_budget` |
| M4 `-m` without PYTHONPATH | `pyproject.toml`, `pip install -e .`, `argus` CLI stub | subprocess `--help` tests |
| M5 concurrent migrators | OS file lock around the whole migration incl. connect (review H-1), plus in-tx re-check | 4 rounds x 3 processes on a fresh file |
| L2, L3, L14 | dead expression removed; run_tool catches Exception; smoke test think=False + timeout | tests / scanner |
| (found by mutation check) | `atomic()` test could not tell BEGIN from BEGIN IMMEDIATE | read-then-write test + SQL trace test; mutant `deferred-begin` |

Backlog (medium/low, not fixed in Phase C): M2, M3, M6, M7, M8, L1, L4-L13,
L15, review L-4 (H1 control is query-level), review L-7. Most of M2/M3/L4/L7
are superseded by the graph engine; see BUILD_STATE.md.
