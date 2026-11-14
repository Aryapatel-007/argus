# Argus

Argus watches web pages you care about (university courses, visa policy,
job postings), notices what genuinely changed, remembers it, and drafts a
message about it that **never leaves your machine until you approve it**.
Everything runs locally: local models through Ollama, one SQLite file, no
paid APIs, no telemetry.

```
argus watch                 # check every target in config/targets.yaml now
argus approvals             # see what Argus wants to send
argus approvals approve 3   # let it go (or: reject 3)
argus dashboard             # http://127.0.0.1:8787 - runs, traces, routing, approvals, health
argus up                    # scheduler + dashboard, unattended
```

## How it works, in two minutes

**A run is a typed task graph, not a fixed loop.** For each target a `plan`
node picks reusable subgraph templates (it never invents raw structure):

```
plan -> fetch_page -> extract -> critique -> resolve_entity -> diff -> write_fact -> remember
                \-> recall (in parallel)                          \
                                                                   summarize -> propose_action -> approval_gate -> execute_action
```

If a field is missing, `extract` emits a trigger and a `replan` node grows
the graph at run time (e.g. follow a same-site link to the fees page).
Retries are new nodes, never back edges; the graph stays a DAG.

**A router picks a local model per node, and it schedules around VRAM.**
The laptop fits one chat model at a time and a swap costs 3-8 seconds, so
every candidate is scored on measured quality, latency, swap cost and
failure risk, against what is resident right now. Quality comes from a
profiler (`scripts/profile_models.py`) and is then learned per (node type,
model) with Thompson sampling, with exploration capped. Every candidate's
score is stored, so "why this model?" is one query (`argus why NODE`).
Strategies per node: single, cascade (cheap model first), cross-model
agreement (required for anything that proposes an external action),
self-consistency, and reflexion retries that feed the critic's reasoning
back in.

**Two lanes.** One GPU slot for model calls, several I/O slots for page
fetches and memory calls, so a page loads while a model thinks.

**Crash-safe by construction.** Every node status change and its checkpoint
event commit in one SQLite transaction. Routing decisions are stored and
frozen before the call; finished samples are stored and reused. A hard kill
at any point resumes without repeating work, without duplicate facts and
without sending anything twice. This is proven by tests that kill real
processes mid-call and then check the process list for orphans.

**The approval gate is structural.** A validator refuses any graph where an
`execute_action` is reachable without passing an `approval_gate`; readiness
waits for a human decision; and the executor claims the approval row before
acting. Property-based tests (hypothesis) try random graphs to bypass it,
with negative controls proving those tests can fail.

**Memory is a separate MCP server** (SQLite + sqlite-vec, embeddings from
`nomic-embed-text`): versioned facts, change feed, hybrid relevance/recency/
importance recall. Pages come from a second MCP server driving headless
Chromium through Playwright. Both are spawned per run over stdio.

**Untrusted text stays fenced.** Page text reaches prompts only inside an
escaped `<untrusted>` block; deterministic validators reject any number a
model writes that is not on the page; the planner cannot fetch off-site.

## Did the complexity pay off?

See [`notes/routing-results.md`](notes/routing-results.md): the same 24
scenarios run three ways (single-model baseline, graph with fixed routing,
graph with learning), with quality, wall-clock time, model swaps and
approval-gate violations reported side by side.

## Setup

Windows 11, Python 3.11, an NVIDIA GPU with ~8 GB VRAM, [Ollama](https://ollama.com).

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
python -m playwright install chromium
ollama pull qwen3.5:9b; ollama pull phi4-mini; ollama pull llama3.1; ollama pull nomic-embed-text
argus migrate
python scripts/profile_models.py      # measure your models (once, ~10 min)
python scripts/demo.py                # a 2-minute offline demo, no internet needed
python -m pytest -q                   # the suite (Chromium + Ollama for a few tests)
```

Configure what to watch in `config/targets.yaml`, which models are allowed
in `config/models.yaml`, and router weights in `config/routing.yaml`.

## Repository map

| Path | What |
|---|---|
| `src/argus/graph/` | graph engine: node types, validator, store, readiness, router, bandit, health, strategies, handlers, engine |
| `src/argus/servers/` | MCP servers: memory (facts + vectors), browser (Playwright) |
| `src/argus/eval/` | scenarios, harness, judge, fixture site |
| `src/argus/dashboard/` | local FastAPI dashboard |
| `src/argus/orchestrator/` | the original Sprint 1 linear loop, kept as a reference |
| `notes/` | measurements, audit, decisions log, Phase G results |
| `ARGUS_BUILD.md`, `BUILD_STATE.md` | the build plan and where it stands |
