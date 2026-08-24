# Design decisions

Why, not what. Entries here are decisions made deliberately, with the
reasoning that led to them, so a later sprint doesn't re-argue something
already settled. See also CLAUDE.md's "Design decisions already made"
section for the Sprint 0/1 decisions this file continues.

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
