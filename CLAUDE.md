# Argus

Local-first autonomous multi-agent monitoring system. Watches a list of targets
(companies, topics, job postings), checks them on a schedule, remembers what it
finds, detects changes, and proposes actions that always wait for human approval.

Solo project, built by a BTech CSE student as a portfolio centerpiece. Built in
sprints, currently Sprint 2.

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
- Windows 11, PowerShell 7.6
- Python 3.11.9
- NVIDIA driver 610.88, CUDA 13.3
- Ollama CLI 0.24.0, `ollama` Python package 0.6.2 — different numbers, don't
  conflate them

**Measured, not assumed.** `qwen3.5:9b` at Q4_K_M needs ~8.2GB total (4.0GB GPU
weights + 2.2GB output layer + 1.4GB KV cache + 0.55GB compute graph) against a
~7.6GB ceiling. Ollama pushes the output layer to CPU. That is one layer of 33,
but a disproportionately heavy one, which is why `ollama ps` reports a ~70/30
GPU/CPU split. Layer count and memory share measure different things; both
readings are correct.

Throughput ~19-21 tok/s on that split. Cold load 6.00s, warm 0.12s.

**Assume VRAM is scarce.** Two chat models never co-reside. Qwen 9B (6.6GB) +
Phi-4-mini (2.5GB) = 9.1GB against 7.6GB. Every alternation costs a 6s reload.

---

## Thinking control — settled, do not rediscover

`qwen3.5:9b` is a reasoning model. Unconstrained it spends nearly all generation
on thinking. Measured (`notes/sprint1_thinking_probe.json`), one-sentence prompt:

- thinking on (default): 1228-1561 eval tokens, 67-83s
- `think=False`: 37 eval tokens, 2.1s

~35x faster, ~97.6% of generation was reasoning.

- `think="low"` is NOT a middle setting here. It is silently coerced to True and
  produces byte-identical results to `think=True`. Thinking is binary on this
  model. Do not use levels.
- `format=` JSON schema composes with `think=False` at zero measured cost.
- One `think=True` run went runaway: 6163 tokens, 1583s, **empty content**, with
  tok/s collapsing to 3.9 under a heavier CPU offload. With thinking on,
  generation length is unbounded and can return nothing at all. A weekly
  scheduled system cannot contain a step that might take 26 minutes and yield
  an empty string.

---

## Architecture

- **Scheduler** — APScheduler, in-process, daily or weekly triggers.
- **Orchestrator** — hand-built plan/act/observe/critique state machine.
  Based on ReAct (reasoning + acting interleaved) and Reflexion (self-critique
  on failure feeds the retry).
- **Tool servers** — custom MCP servers built with the official Python SDK:
  - Memory server: `remember_fact`, `recall_facts`, `list_changes_since`
  - Browser server: `search`, `open_page`, `extract_structured`
- **Models via Ollama:**
  - Planner / critic / writer / judge: `qwen3.5:9b`
  - `phi4-mini`: pulled, in the architecture, unused until Sprint 7 gives real
    routing data. At 2.1s per call a swap to it costs 6s to save ~1s.
  - Embeddings: `nomic-embed-text`, 768 dimensions (fixes sqlite-vec column
    width in Sprint 2)
- **Storage** — single SQLite file, WAL mode, sqlite-vec extension for vectors.

### Core loop, per target, per run

1. Plan what to check
2. Act — call a tool
3. Observe the real result
4. Critique — good enough? If not, write a reflection and retry the act step
5. Externally visible? If no, save to memory. If yes, queue for approval and wait

Every state transition is checkpointed to SQLite immediately, in one transaction,
so a crash resumes from the last completed step. Side-effecting actions carry an
idempotency key generated and stored *before* the action runs.

### Schema

- `runs` — one row per scheduled execution
- `tasks` — sub-tasks the planner generated within a run
- `checkpoints` — state after every transition
- `llm_calls` — every Ollama call with latency, tok/s, token counts
- `facts` — versioned knowledge per target (Sprint 2, migration 002)
- `approvals` — pending and resolved human decisions (Sprint 2, migration 002)

---

## Design decisions already made — don't relitigate these

- **Hand-built state machine before LangGraph.** Deliberate. The point is
  understanding what a checkpointer and idempotent retry actually solve. Do not
  suggest migrating to LangGraph during early sprints.
- **Kept `qwen3.5:9b` despite the partial CPU offload.** Do not propose dropping
  to 4b. The latency problem was thinking tokens, not model size or inference
  speed. Smaller models are worse at structured output, which is exactly what
  the planner and critic depend on.
- **Single model for all roles in Sprint 1.** Zero swaps. Model bindings per role
  come from config, so routing to `phi4-mini` later is a config change, not a
  refactor.
- **Local models over hosted APIs.** Accepted tradeoff: local models are worse at
  structured output. That is *why* deterministic validation and retry logic wrap
  every tool call and judge call.
- **Custom MCP servers, not just consuming existing ones.** Deliberate skill
  investment plus no existing server fits the memory design.
- **sqlite-vec over Chroma.** Embedded, no separate process, fact count stays in
  the thousands.
- **Browser Use layered on Playwright**, not hand-coded selectors. Page layouts
  vary across dozens of uncontrolled sites. Raw Playwright stays available as a
  fallback for specific hand-tuned pages.

---

## Project layout

```
argus/
  config/          targets.yaml — what Argus watches
  data/            SQLite file (gitignored)
  notes/           sprint measurements and findings
  scripts/         one-off runners, smoke tests, probes
  src/argus/
    orchestrator/  the plan-act-critique state machine
    servers/       MCP servers (memory, browser)
    storage/       schema, migrations, sqlite-vec setup
  tests/
```

---

## Conventions

- Python 3.11, plain `venv` at `.venv/`, activated before any work.
- Install dependencies only in the sprint that needs them. Do not pre-install
  MCP SDK, sqlite-vec, Browser Use, or APScheduler ahead of time — version drift
  before first use is a real cost.
- Pin `mcp>=2.0,<3`. Re-examined in Sprint 2: v2 is the stable line as of
  2026-07-28 and `pip install mcp` resolves to it. v1 is maintenance-only.
  The port from v1 course material is small for a decorator-built server:
  `from mcp.server import MCPServer` replaces
  `from mcp.server.fastmcp import FastMCP`; `@mcp.tool()` is unchanged.
- MCP tools are always `async def`. In v2 a sync `def` tool runs on a worker
  thread, and `sqlite3.Connection` is thread-affine.
- A stdio MCP server owns stdout. All logging goes to stderr or a file.
  Never `print()` in a server module.
- Load sqlite-vec with `sqlite_vec.load(conn)`, never
  `conn.load_extension()`. On Windows `loadable_path()` returns a path with
  no extension and that bare path does not exist; `load()` resolves it.
- `think` is a TOP-LEVEL field on Ollama chat/generate, not inside `options`.
  Pass `think=False` explicitly on EVERY call, every role. Never rely on the
  default.
- Never use `num_predict` as a thinking cap. With thinking on it truncates
  mid-reasoning and returns no answer.
- Every Ollama call needs a hard 120s timeout and must log tok/s. A degraded
  CPU offload has been observed dropping throughput 5x mid-session.
- The state transition and its checkpoint insert go in ONE transaction.
- Async by default for anything I/O-bound. Use `httpx.AsyncClient`, never
  blocking `requests` inside an `async def`.
- Model tags come from config, never hardcoded in source.
- Log structured rows, not print statements. Every plan, every tool call with
  latency and token count, every critic verdict.
- On Windows, `python` not `python3`.

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

Table ownership: orchestrator owns runs, tasks, checkpoints. Memory server
owns facts, facts_vec. Both write llm_calls — the memory server logs its
embedding calls with role='embedder'. The write lock is database-wide, not
per-table; ownership is for clarity, not for concurrency.

Only the orchestrator runs migrations. The memory server reads
PRAGMA user_version at startup and exits with a clear error on mismatch.

Load sqlite-vec on every connection that opens the file, including the
orchestrator's.

---

## How to work with me

- I build first and learn as I go. Show me runnable code over explanation.
- Explain *where and when* something is used in Argus, not abstract theory.
- Short sentences. Plain language. Complete, copy-pasteable code blocks.
- If you are unsure of exact syntax, a version number, or an API detail, say so
  and point me at official docs. Do not guess and do not invent APIs.
- Don't re-teach LangChain, LangGraph, MCP, Playwright, or async Python at a
  beginner level — I know these.
- Confirm each step worked before moving to the next.
- Design decisions get made in a separate architect chat, not here. If you hit a
  design question this file doesn't answer, stop and tell me rather than
  deciding it yourself.

---

## Current state — Sprint 2

Sprint 1 closed. Hand-built plan/act/observe/critique state machine,
SQLite-checkpointed after every transition, crash-resume verified against
real Ollama. Migration chain at 001c.

Sprint 2 goal: the memory server as a real custom MCP server, SQLite plus
sqlite-vec, tested standalone before the orchestrator touches it.

Decisions made at the top of Sprint 2:
- MCP SDK v2, replacing the earlier `mcp<2` pin.
- Embeddings are generated inside the memory server, not the orchestrator.
  Tool surface stays semantic (`query: str`), write/query symmetry is
  guaranteed by construction, and no 768-float arrays cross stdio.
- The memory server is spawned per run and receives `--run-id`, so embedding
  calls are attributable in llm_calls without polluting the model-facing
  tool schema.
- Idempotency key lives on `approvals`, not `tasks`. It must survive across
  runs; a task does not. `tasks.idempotency_key` is dropped in 002.
- Retrieval ships relevance + recency at 0.7/0.3. The `importance` column
  exists in 002 but stays NULL until Sprint 5, treated as a neutral constant
  by the scorer.
- Entity resolution in Sprint 2 is exact `entity_key` match only. Fuzzy
  matching is Sprint 4.

Measured in Sprint 2 Step 0:
- qwen3.5:9b and nomic-embed-text ARE co-resident. Embeddings run inline per
  fact. No batching, no swap cost.
- sqlite-vec 0.1.9 loads cleanly on Windows via sqlite_vec.load().
- Node 24.11.1, past the 22.19.0 floor for MCP Inspector v2.

Open and unverified:
- `think=False` quality on writer-shaped prompts. Probe before wiring.
- Co-residency was measured with nothing else running. Sprint 3 adds a
  browser; re-measure then.
- Single-shot planner. Sprint 4. Do not fix here.