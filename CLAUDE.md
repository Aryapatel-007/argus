# Argus

Local-first autonomous multi-agent monitoring system. Watches a list of targets
(companies, topics, job postings), checks them on a schedule, remembers what it
finds, detects changes, and proposes actions that always wait for human approval.

Solo project, built by a BTech CSE student as a portfolio centerpiece. Built in
sprints, currently Sprint 0.

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

**Measured, not assumed:** `qwen3.5:9b` at Q4_K_M needs ~8.2GB total (4.0GB GPU
weights + 2.2GB output layer + 1.4GB KV cache + 0.55GB compute graph). This does
not fit in 7.6GB. Ollama pushes the output layer to CPU, producing a ~70/30
GPU/CPU split and ~70s response times. Reducing context from 4096 to 2048 did
not meaningfully help — the bottleneck is weights, not KV cache.

Open decision (Sprint 1): keep `qwen3.5:9b` with the CPU split, or drop to
`qwen3.5:4b` (3.4GB) which fits cleanly. Untested lever: `OLLAMA_KV_CACHE_TYPE=q4_0`.

**Assume VRAM is scarce.** Never assume two chat models can be resident
simultaneously. Qwen 9B (6.6GB) + Phi-4-mini (2.5GB) = 9.1GB, which never fits.

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
  - Planner / writer / judge: `qwen3.5:9b` (see open decision above)
  - Fast critic / router: `phi4-mini`
  - Embeddings: `nomic-embed-text`
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
- `facts` — versioned knowledge per target, indexed on target_id, run_id, created_at
- `approvals` — pending and resolved human decisions, including the reasoning

---

## Design decisions already made — don't relitigate these

- **Hand-built state machine before LangGraph.** Deliberate. The point is
  understanding what a checkpointer and idempotent retry actually solve. Do not
  suggest migrating to LangGraph during early sprints.
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
  scripts/         one-off runners, smoke tests
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
- Pin `mcp < 2` when the MCP SDK does get installed.
- Async by default for anything I/O-bound. Use `httpx.AsyncClient`, never
  blocking `requests` inside an `async def`.
- Model tags come from config, never hardcoded in source.
- Log structured rows, not print statements. Every plan, every tool call with
  latency and token count, every critic verdict.
- On Windows, `python` not `python3`.

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

---

## Current state — Sprint 0

Goal: environment setup and scaffolding, ending with one successful Python call
to a local model through Ollama.

Done:
- NVIDIA driver updated 555.97 → 610.88, fixing total Ollama GPU discovery failure
- Ollama confirmed at `library=CUDA`, 7.6GB available
- `qwen3.5:9b` and `nomic-embed-text` present locally
- VRAM ceiling measured (see Hardware above)

Remaining:
- Pull `phi4-mini`
- Project scaffold + venv + `ollama` and `pyyaml` installed
- `config/targets.yaml`
- `scripts/smoke_test.py` — chat reply with latency and token count, embedding
  vector length printed, both model tags read from config
- Notes: `ollama ps` output, measured model swap time, whether Qwen and nomic
  can co-reside

The embedding vector length matters beyond Sprint 0 — it fixes the sqlite-vec
column width in Sprint 2.
