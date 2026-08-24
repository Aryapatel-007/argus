# Design decisions

Why, not what. Entries here are decisions made deliberately, with the
reasoning that led to them, so a later sprint doesn't re-argue something
already settled.

## Sprint 0/1

## Hand-built the orchestration loop before adopting LangGraph

Chose to write the plan, act, critique state machine by hand in Sprint 1, rather than starting directly with LangGraph, even though I already knew LangGraph practically from a prior course. A framework hides exactly the hard parts worth understanding first: what a checkpointer actually solves, what happens when a step fails partway through, what idempotent retries protect against. Building it raw first means any later use of LangGraph is informed rather than cargo-culted. LangGraph remains an option to migrate to afterward, once the underlying mechanics are already understood firsthand.

## Local models instead of a paid API

Chose Ollama running local, quantized models (Qwen3.5-9B and Phi-4-mini) instead of calling a hosted API. This keeps the project at zero recurring cost, keeps all data on the machine rather than sent to a third party, and matches the actual hardware available (RTX 4060, 8GB VRAM). The tradeoff, accepted deliberately: local models are less reliable at structured output than frontier hosted models, which is exactly why deterministic validation and retry logic exist ahead of every judge or tool call.

## Custom-built MCP servers instead of only consuming existing ones

Chose to build the memory and browser tool servers from scratch using the official MCP Python SDK, rather than only wiring together pre-built third-party MCP servers. No ready-made MCP server does what this specific memory design (recency, importance, and relevance-weighted retrieval, tied to sqlite-vec) or this specific browsing job needs. Building them from scratch is also a deliberate skill investment, since almost nobody at a junior level has actually built an MCP server rather than only consumed one.

## sqlite-vec instead of a heavier vector database

Chose sqlite-vec, an embedded extension living inside the same SQLite file as everything else, over a separate vector database like Chroma. At single-developer scale, with a fact count that will realistically stay in the thousands, an embedded store needs no server process, no separate deployment, and no second system to keep in sync with the primary database. Chroma remains the better choice only once collections, metadata filtering at scale, or a genuinely large corpus are actually needed, none of which apply here.

## Browser Use layered on top of Playwright, rather than hand-coded Playwright selectors alone

Originally planned to hand-write Playwright automation directly, reusing prior internship experience. Reconsidered this once the actual failure mode became clear: hand-coded selectors are brittle against page layouts that vary across dozens of different, uncontrolled company websites, and that can silently change over time. Browser Use adds an LLM-driven layer that interprets a page rather than matching a fixed selector, while still running on Playwright underneath, so the original skill isn't wasted, only the approach of hand-coding every single target site is dropped. Raw Playwright remains available as a fallback for any one specific page worth hand-tuning.

## Kept the 9B planner model even though it doesn't fit entirely in VRAM

Sprint 0 measurement showed Qwen3.5-9B at Q4_K_M needs roughly 8.2GB in total against a usable ceiling of about 7.6GB, so it runs with one of its 33 layers held back on the CPU instead of fully offloaded to the GPU. The obvious response was to drop to a 4B-class model that fits cleanly. Chose not to. Measured throughput with the partial offload is 24.3 tok/s, which is comfortable for a system that wakes up on a weekly schedule rather than serving live chat. The slow first response that suggested a problem turned out to be one long generation, not slow inference: the model spent roughly 1200 tokens reasoning to produce a twenty-word answer. The real latency lever is therefore the thinking budget on routine calls, not the parameter count, and switching to a smaller model would have traded away planning quality to fix a problem it wasn't causing. Phi-4-mini stays in the architecture for the frequent cheap calls exactly as originally planned, which is the right job for a smaller model, rather than as a replacement for the planner.

## Thinking is disabled on every call, and the planner stayed on the 9B model

Qwen3.5 is a reasoning model, and Sprint 1's probe showed it spending 1228-1561 eval tokens and 67-83 seconds to produce a one-sentence answer. With `think=False` the same prompt returned 37 tokens in 2.1 seconds. Roughly 97.6% of generation was reasoning the system never used. Two things came out of that measurement. First, `think="low"` is not a middle gear on this model, it is silently coerced to on and returns byte-identical output to `think=True`, so thinking here is binary and the SDK's type hint is misleading. Second, and more important, one thinking-enabled run went runaway: 6163 tokens over 26 minutes, returning empty content, with throughput collapsing to 3.9 tok/s under a heavier CPU offload. With thinking on, generation length is unbounded and can terminate with no answer at all. A system that wakes on a weekly schedule and runs unattended cannot contain a step that might take 26 minutes and produce nothing, so `think=False` is passed explicitly on every call rather than relying on any default.

This also closed the model question. At 2.1 seconds per call the 9B model is comfortably fast enough for planner, critic, and writer, so Sprint 1 runs entirely on one model with zero swaps. Routing to Phi-4-mini would cost a 6 second reload to save roughly a second, which is a losing trade until Sprint 7 produces real quality data to justify it. Model bindings per role come from config rather than source, so that routing change stays a config edit rather than a refactor.

One risk stays open and is deliberately recorded rather than assumed away: `think=False` was verified only on a trivial prompt. Whether it degrades output on planner-shaped prompts, a goal plus tool schemas producing a decomposed task list, is untested. If it does, the fallback is not to re-enable thinking but to request a `reasoning` field inside the structured output schema, which is bounded, inspectable, and loggable, unlike opaque thinking tokens. Sprint 6's observability requirements want that reasoning logged regardless.

## The human approval gate is a hard rule, not a confidence threshold

Any action visible outside the machine queues for human approval regardless of the model's confidence score. Confidence is allowed to decide whether an internal fact gets double-checked before being saved. Confidence is never allowed to decide whether an external action skips human review. This is a deliberate, non-negotiable line, not a tunable parameter.

## Sprint 2

## MCP SDK v2 instead of the v1 pin

The pin was set when v2 had just landed and the course material Argus was
built against was still v1. That was the right call at the time — a just-shipped
major version on a dependency this central wasn't worth the risk. Re-examined
deliberately at the top of Sprint 2 rather than carried forward by inertia: v2
is now the stable line, `pip install mcp` resolves to it by default, and v1 is
maintenance-only.

The port turned out to cost almost nothing for a decorator-built server — one
import changes (`from mcp.server import MCPServer` instead of
`from mcp.server.fastmcp import FastMCP`), `@mcp.tool()` is unchanged. v2's
in-memory `Client(mcp)` also gives a real pytest story that v1 didn't have,
which matters more than the import: it means MCP Inspector is confirmation,
not the only way to exercise a tool. Pinning to v1 here would have meant
porting later, inside whatever sprint first needed the in-memory client,
already mid-way through debugging something unrelated.

## Embeddings are generated inside the memory server

The losing argument first, because it's a fair one: the orchestrator already
owns the Ollama client, the model config, and the `llm_calls` logging. Keeping
the memory server pure SQLite — no network calls, no Ollama dependency — would
make it simpler and fully deterministic, and it's the more obvious split of
responsibility on paper.

It lost for three reasons. First, the tool surface has to be semantic
(`query: str`, not `query_embedding: list[float]`) because the caller is the
planner LLM, not application code — an LLM cannot produce a 768-float array as
a tool argument. Second, write and query embeddings must come from the same
model by construction, not by discipline enforced across two call sites; if
the embedding client lived in the orchestrator, nothing would stop a future
change from calling it with a different model tag on one path than the other.
Third, passing 768 floats as JSON over stdio is ~12KB per call for no reason,
when the vector never needs to leave the process that's about to write it into
`sqlite-vec`. None of this costs anything extra in resources — Ollama is a
shared server process, so putting the client in the subprocess doesn't
duplicate VRAM or add a load. And the observability objection — "then
`llm_calls` misses embedding calls" — is answered directly: the memory server
writes its own `llm_calls` rows with `role='embedder'`, attributed to a run via
a `--run-id` passed at spawn, so nothing is lost, it's just logged from a
different process.

## The idempotency key lives on approvals, not tasks

001b put `idempotency_key` on `tasks`. Sprint 5's plan, written separately,
put it on `approvals`. Two candidate homes for the same key is itself the bug
— it means the key's actual job was never pinned down.

The deciding argument is lifetime, not convenience. An approval is queued in
one run and can be executed in a later one, once a human acts on it — that gap
is exactly the window an idempotency key exists to protect, so a retried or
delayed execution doesn't double-fire the external action. A task is
run-scoped; it doesn't survive past the run that created it, so a key on
`tasks` can't cover the window that matters. Cardinality fails too: one task
can produce zero side effects (a read) or, given retries, more than one
attempt at a side effect — the key needs to track the side effect, not the
task that requested it. `tasks.idempotency_key` is dropped in migration 002
rather than left in place unused, because a dead column with the right name in
the wrong place is exactly what makes Sprint 5 re-open this argument instead of
finding it already settled.

## No embedding-prefix on fact content

nomic-embed-text documents `search_query:` / `search_document:` task prefixes
as improving retrieval. Tested against plain unprefixed text on real short
facts before adopting them, rather than assuming the documented default is
correct for this content. It measurably wasn't: symmetric `search_document:`
prefixing on both sides of a similar/unrelated comparison **inverted** the
margin (similar pair scored *lower* than the unrelated pair, −0.0142), while
no prefix gave the only correct-direction result of the three forms tried
(+0.0209).

Shipping unprefixed. Revisit with a larger measured set — three strings is
enough to catch an inversion, not enough to certify a retrieval strategy — if
Sprint 6/7 recall quality looks weak.
