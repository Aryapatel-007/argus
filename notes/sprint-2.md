# Sprint 2 — memory server

## Step 0: environment probe

`scripts/probe_sprint2.py`, run on the real dev machine (Windows 11,
Python 3.11.9, `.venv`). 19/19 checks passed. Actual summary output, verbatim:

```
=== SUMMARY ===
[PASS] sqlite_version >= 3.41 (sqlite-vec recommendation)
[PASS] sqlite_version >= 3.35 (ALTER TABLE DROP COLUMN)
[PASS] sqlite3.Connection has enable_load_extension
[PASS] sqlite-vec loads
[PASS] 768-dim KNN round trip
[PASS] PRAGMA journal_mode reports 'wal'
[PASS] WAL two-writer: 400 rows landed, zero sqlite3.OperationalError
[PASS] call 1 (planner chat) succeeded
[PASS] call 2 (embedding) succeeded
[PASS] call 3 (planner chat) succeeded
[PASS] planner+embed co-resident (call 3 paid no cold load) -> CO-RESIDENT
[PASS] embedding vector length == 768
[PASS] node.js present
[PASS] node >= 22.19.0 (MCP Inspector v2 requirement)
[PASS] npx present
[PASS] git log succeeded
[PASS] git status succeeded
[PASS] notes/ exists
[PASS] prompts/ exists (src/argus/orchestrator/prompts)
[PASS] migrations/ exists (src/argus/storage/migrations)
```

### sqlite-vec loads via sqlite_vec.load(conn), first attempt

`sqlite-vec` 0.1.9 loaded on the first attempt via `sqlite_vec.load(conn)`.
The manual fallback chain (`conn.load_extension(loadable_path())`, then
`+ ".dll"`, then `os.add_dll_directory()` + retry) never ran — it wasn't
needed.

`sqlite_vec.loadable_path()` returns a path with **no extension**
(`...\sqlite_vec\vec0`), and that bare path does not exist on disk. Only
`vec0.dll` does. `sqlite_vec.load(conn)` resolves the platform extension
internally; a hand-rolled `conn.load_extension(sqlite_vec.loadable_path())`
would fail on Windows without appending `.dll` first.

**Rule:** always call `sqlite_vec.load(conn)`. Never call
`conn.load_extension()` directly against `loadable_path()`.

### Ollama verdict: CO-RESIDENT

Three calls, same process, same `ollama serve`:

| call | model | wall clock | load_duration |
|---|---|---|---|
| 1 | `qwen3.5:9b` (planner) | 6.364s | 6.156s — cold load |
| 2 | `nomic-embed-text` (embed) | 0.896s | — |
| 3 | `qwen3.5:9b` (planner) | 0.265s | 0.136s — warm, no reload |

`ollama ps` after each call:

- After call 1: `qwen3.5:9b` alone, 8.7GB, 29%/71% CPU/GPU split.
- After call 2: `qwen3.5:9b` (8.7GB, 29%/71% CPU/GPU) **and**
  `nomic-embed-text:latest` (595MB, 100% GPU) resident together.
- After call 3: both still resident, unchanged.

Call 3's 0.136s load_duration versus call 1's 6.156s is the cold-vs-warm
signature from CLAUDE.md's thinking-control measurements — call 3 paid none
of it. The two models fit in VRAM at the same time.

**Consequence:** embeddings can be generated inline, per fact, inside the
memory server's tool calls. No batching, no queuing writes to embed later, no
model-swap cost to reason about. This directly enables the Sprint 2 decision
to put the Ollama embedding client inside the memory server rather than the
orchestrator — see `design-decisions.md`.

### WAL two-writer: 400 rows, zero OperationalError

Two OS subprocesses, one shared WAL-mode SQLite file, 200
`BEGIN IMMEDIATE; INSERT; COMMIT` iterations each. All 400 rows landed, zero
`sqlite3.OperationalError` across both workers. Elapsed wall-clock: 0.280s.

### Toolchain versions

- `mcp` resolved to **2.0.0** (pulls in `mcp-types` **2.0.0**).
- `sqlite-vec` resolved to **0.1.9**.
- Node **24.11.1** — past the 22.19.0 floor for MCP Inspector v2.
- Embedding dimension confirmed **768** (`nomic-embed-text`), matching the
  `FLOAT[768]` column width `sqlite-vec` will use.
