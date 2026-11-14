# Argus

Local-first autonomous multi-agent monitoring system. Watches a list of targets
(companies, topics, job postings), checks them on a schedule, remembers what it
finds, detects changes, and proposes actions that always wait for human approval.

Solo project, built by a BTech CSE student as a portfolio centerpiece.
Sprints are retired. The build now runs in phases defined in `ARGUS_BUILD.md`.
**Every session: read `BUILD_STATE.md` first, then this file, then
`ARGUS_BUILD.md`.**

---

## Hard constraints — never violate these

1. **Everything runs locally.** No paid APIs, no hosted inference, no cloud
   services at runtime. Ollama with local quantized models only.
2. **Zero recurring cost.** If a solution requires a subscription or per-call
   billing, it is out of scope. Propose a local alternative instead.
3. **All data stays on the machine.** No telemetry, no external logging, no
   sending target data anywhere.
4. **Nothing externally visible happens without human approval.** Any action
   that leaves the machine writes a row to the `approvals` table and blocks.
   This is a hard rule, not a confidence threshold.

If a request seems to conflict with these, stop and ask rather than assuming an
exception is intended.

---

## Hardware — this shapes real decisions

- Asus ROG Strix G16, i7-13650HX (14 cores / 20 threads)
- 16GB system RAM, typically 5GB free
- RTX 4060 Laptop, 8GB VRAM, ~7.6GB usable
- Windows 11, PowerShell 7.6, Python 3.11.9
- NVIDIA driver 610.88, CUDA 13.3
- Ollama CLI 0.24.0, `ollama` Python package 0.6.2 — different numbers, don't
  conflate them. Live versions: `notes/environment.md`.

**Measured, not assumed.** `qwen3.5:9b` at Q4_K_M needs ~8.2GB total (4.0GB GPU
weights + 2.2GB output layer + 1.4GB KV cache + 0.55GB compute graph) against a
~7.6GB ceiling. Ollama pushes the output layer to CPU: one layer of 33, but a
heavy one, which is why `ollama ps` reports a ~70/30 GPU/CPU split. Throughput
~19-21 tok/s on that split. Cold load 6.00s, warm 0.12s.

**Assume VRAM is scarce.** Two chat models never co-reside. Qwen 9B (6.6GB) +
Phi-4-mini (2.5GB) = 9.1GB against 7.6GB. Every alternation costs a reload.
The graph router treats swap cost as a first-class term (ARGUS_BUILD.md 6.2).

---

## Thinking control — settled, do not rediscover

`qwen3.5:9b` is a reasoning model. Unconstrained it spends nearly all generation
on thinking. Measured (`notes/sprint1_thinking_probe.json`), one-sentence prompt:

- thinking on (default): 1228-1561 eval tokens, 67-83s
- `think=False`: 37 eval tokens, 2.1s

- `think="low"` is NOT a middle setting here. It is silently coerced to True.
  Thinking is binary on this model. Do not use levels.
- `format=` JSON schema composes with `think=False` at zero measured cost.
- With thinking on, generation length is unbounded: one run took 1583s and
  returned **empty content**. The writer probe (Phase C) saw 5 of 7 thinking
  runs exceed a 300s kill. `think=False` on every call, every role, every model.

---

## Architecture

- **Scheduler** — APScheduler in-process (`argus scheduler` / `argus up`):
  per-target cron, catch-up, recovery of hard-killed runs, weekly reflect.
- **Graph engine** (`src/argus/graph/`) — typed, dynamically expanding task
  graph: validator (gate on every path, DAG, limits), store (every status
  change + its node_events checkpoint in one tx), readiness, router (measured
  priors + Thompson sampling, swap-aware), strategies, two-lane engine,
  procedural memory. Spec: `notes/specs/graph-engine.md`. The Sprint 1 loop
  (`orchestrator/`) is kept as a reference (D-024). No agent framework.
- **Tool servers** — custom MCP servers built with the official Python SDK,
  spawned per run over stdio (`graph/mcp_clients.py`):
  - Memory server: `remember_fact`, `recall_facts`, `list_changes_since`,
    `list_current_facts`
  - Browser server: raw Playwright (D-002), page text + links out, no
    selectors, internal origins blocked, degrades instead of raising
- **Product** — `app.py` (sessions, watch, approvals), `observe.py` (traces,
  routing explanations), `dashboard/` (127.0.0.1), `eval/` (24 scenarios,
  judge, harness), `cli.py` (`argus ...`).
- **Models via Ollama:** a measured pool, not fixed tiers. Which models are
  allowed lives in config; what they are good at lives in the database
  (`model_profiles`, `arm_stats`). Embeddings: `nomic-embed-text`, 768 dims.
- **Storage** — single SQLite file, WAL mode, sqlite-vec extension for vectors.

Every state transition is checkpointed to SQLite in the same transaction as
the transition, so a crash resumes from the last completed step. Side-effecting
actions carry an idempotency key on `approvals`, generated and stored *before*
the action runs.

---

## Design decisions already made — don't relitigate these

Full reasoning: `design-decisions.md` (Sprints 0-2) and `notes/decisions-log.md`
(numbered, Phase A onward).

- **Hand-built orchestration.** No LangGraph, CrewAI, AutoGen or any agent
  framework. Narrow libraries (MCP SDK, sqlite-vec, Playwright, APScheduler)
  are fine.
- **Kept `qwen3.5:9b` despite the partial CPU offload.** The latency problem
  was thinking tokens, not model size.
- **Local models over hosted APIs.** Local models are worse at structured
  output, which is *why* deterministic validation and retry wrap every call.
- **Custom MCP servers**, not only consuming existing ones.
- **sqlite-vec over Chroma.** Embedded, no separate process.
- **Raw Playwright, not Browser Use** (D-002, supersedes the Sprint 0 entry).
- **Idempotency key lives on `approvals`**, not tasks.
- **Embeddings are generated inside the memory server.**

---

## Project layout

```
argus/
  ARGUS_BUILD.md   the build directive (phases A-G)
  BUILD_STATE.md   where the build is; read first every session
  config/          targets.yaml — what Argus watches
  data/            SQLite file (gitignored)
  notes/           measurements, audit, decisions-log.md
  scripts/         one-off runners, smoke tests, probes, crash tests
  src/argus/
    graph/         engine, router, strategies, handlers, prompts, MCP clients
    servers/       MCP servers (memory, browser)
    storage/       schema, migrations 001-007, leases, sqlite-vec setup
    eval/          scenarios, harness, judge, fixture site
    dashboard/     FastAPI + Jinja2, local only
    orchestrator/  Sprint 1 linear loop (reference)
  tests/           pytest; real-kill crash tests; hypothesis properties
```

---

## Conventions

- Python 3.11, plain `venv` at `.venv/`, activated before any work.
- **All dependencies are installed and pinned up front** in `requirements.txt`
  and `requirements-dev.txt` (D-006). Add a pin when adding a dependency.
- Tests: `python -m pytest` (pytest.ini sets `pythonpath = src`). Older suites
  are unittest classes; pytest collects them. The passing count never goes
  down; it is recorded in BUILD_STATE.md at every commit.
- `mcp` is the v2 line. `from mcp.server import MCPServer`, `@mcp.tool()`.
- MCP tools are always `async def`. In v2 a sync `def` tool runs on a worker
  thread, and `sqlite3.Connection` is thread-affine.
- A stdio MCP server owns stdout. All logging goes to stderr or a file.
  Never `print()` in a server module.
- Load sqlite-vec with `sqlite_vec.load(conn)`, never
  `conn.load_extension()`. On Windows `loadable_path()` returns a path with
  no extension and that bare path does not exist; `load()` resolves it.
- `think` is a TOP-LEVEL field on Ollama chat/generate, not inside `options`.
  Pass `think=False` explicitly on EVERY call. Never rely on the default.
- Never use `num_predict` as a thinking cap.
- Every Ollama call needs a hard 120s timeout and must log tok/s, including
  calls that time out. Throughput has been seen dropping 5x mid-session.
- The state transition and its checkpoint insert go in ONE transaction.
- Every write goes through `db.atomic()` (BEGIN IMMEDIATE). No second helper.
  One named exception: `ollama_client.log_call` is a single autocommit
  INSERT, outside the step transaction on purpose, so the record of a call
  survives a rolled-back step (audit 4, "checked and fine").
- Async by default for anything I/O-bound. `httpx.AsyncClient`, never
  blocking `requests` inside an `async def`.
- Model tags come from config, never hardcoded in source.
- Log structured rows, not print statements.
- On Windows, `python` not `python3`. Child processes are NOT killed with
  their parent; kill and verify by process listing.
- Never commit Co-Authored-By trailers or Claude attribution (global rule).

---

## Storage concurrency — two writers, one file

The orchestrator and the memory server both write to the same SQLite file.
WAL permits one writer at a time, not one writer process. Three rules,
non-negotiable, on every connection in both processes:

1. `PRAGMA busy_timeout=5000` immediately after connect.
2. `BEGIN IMMEDIATE` for any transaction that will write. `busy_timeout`
   does not apply when a deferred transaction upgrades from read to write —
   that returns SQLITE_BUSY instantly regardless of timeout. Python's
   sqlite3 defaults to deferred, so use `isolation_level=None` and issue
   `BEGIN IMMEDIATE` explicitly.
3. Never hold a transaction open across an Ollama call. Embed first,
   then write.

Verified in Sprint 2: two processes, 200 BEGIN IMMEDIATE writes each,
400 rows landed, zero OperationalError.

Table ownership: orchestrator owns runs, tasks, checkpoints and the graph
tables. Memory server owns facts, facts_vec. Both write llm_calls — the memory
server logs its embedding calls with role='embedder'. The write lock is
database-wide; ownership is for clarity, not for concurrency.

Only the orchestrator runs migrations. The memory server reads the
schema_migrations table at startup and exits with a clear error unless its
required migration is applied. It never runs migrations itself.

Migration state lives in the `schema_migrations` table — one row per applied
filename. `PRAGMA user_version` is NOT used and reads 0 even on a fully
migrated database. Guarded by a test.

Load sqlite-vec on every connection that opens the file.

---

## How to work with me

- I build first and learn as I go. Show me runnable code over explanation.
- Explain *where and when* something is used in Argus, not abstract theory.
- Short sentences. Plain language. Complete, copy-pasteable code blocks.
- If you are unsure of exact syntax, a version number, or an API detail, say so
  and check official docs. Do not guess and do not invent APIs.
- Don't re-teach LangChain, LangGraph, MCP, Playwright, or async Python at a
  beginner level — I know these.
- Confirm each step worked before moving to the next.
- **You make design decisions yourself** (D-001). Record each one in
  `notes/decisions-log.md`: chosen, rejected, why. Stop and write
  `BLOCKED: needs Arya` at the top of BUILD_STATE.md only for the cases in
  ARGUS_BUILD.md section 1.4 (hard-constraint conflict, touching real data in
  data/argus.db, force push or history rewrite, credentials or paid accounts,
  disk under 15GB before pulls, Phase G graph engine losing to baseline).

---

## Current state

See `BUILD_STATE.md`. Sprint 1 and Sprint 2 history: `notes/sprint-1.md`,
`notes/sprint-2.md` (its last section is the Sprint 2 closing state).
