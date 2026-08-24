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

## Step 3: migration 002 and the storage layer

Migration `002_facts_approvals.sql`: `facts`, `facts_vec`, `approvals`, plus a
rebuild of `llm_calls`. Transactional (no `-- migrate: no-transaction` marker).

### BEGIN IMMEDIATE — the bug, measured

`db.atomic()` issued a plain `BEGIN`. It now issues `BEGIN IMMEDIATE`. This is
a correctness fix, not a style preference, and the difference was measured
directly rather than argued from the docs.

Two connections, one thread, a write lock deliberately held by a third,
`busy_timeout = 1000` on both:

```
DEFERRED upgrade raised after 0.0000s: database is locked
IMMEDIATE raised after 1.1610s: database is locked
```

**Why this matters.** A plain `BEGIN` starts a DEFERRED transaction, which
takes no lock until its first statement. A transaction that reads and then
writes therefore has to UPGRADE a read lock to a write lock partway through,
and **`busy_timeout` does not apply to that upgrade** — SQLite returns
SQLITE_BUSY instantly, whatever the timeout says, because waiting could
deadlock two readers both trying to upgrade. The 1000ms timeout was set and
completely ignored: 0.0000s.

`busy_timeout` is not the safety net it is usually assumed to be. It only
protects a transaction that took its write lock up front. Under a single
writer this is invisible; it would have surfaced only once the memory server
and the orchestrator wrote concurrently, as intermittent `OperationalError`
under load.

Committed as a matched pair in `test_storage.py` (`TestBeginImmediate`), so
"we use BEGIN IMMEDIATE" is an assertion rather than a docstring claim. Fully
deterministic — a held lock, not scheduling luck.

Audited before changing it: **zero read-only callers of `atomic()`**. Every
call site (`loop.py:413`, `migrate.py:100`, and the tests) unconditionally
writes, so taking the write lock earlier serializes nothing that was
previously concurrent. Reads go through `db.connect()` in autocommit and are
untouched.

### vec0 metadata / partition columns at 0.1.9

**The docs say nothing.** Both https://alexgarcia.xyz/sqlite-vec/ and its API
reference are marked "🚧 This documentation is a work-in-progress 🚧". The
reference names "Metadata Columns", "Partition Keys" and "Auxiliary Columns"
but the constructor and KNN sections are literal `[Constructor](#TODO)` and
`[#### KNN Queries](#TODO)` stubs.

**The extension supports them anyway.** Probed against the installed 0.1.9:

```
OK  bare                        OK  partition key  (target_id TEXT PARTITION KEY)
OK  named INTEGER PRIMARY KEY   OK  auxiliary column (+source_url TEXT)
OK  metadata col (TEXT / INTEGER)
metadata-filtered KNN: [(1, 0.0), (3, 4.0)]   -- filtering genuinely works
```

**002 uses the plain form regardless**: `fact_id INTEGER PRIMARY KEY,
embedding FLOAT[768]`. Undocumented syntax is not a thing to put in a schema —
it can change without a note in a changelog that does not describe it in the
first place. Filtering by `target_id` is a join against `facts`. Revisit if
the syntax gets documented.

### Column types found in the existing schema

- `runs.run_id` — **TEXT** (a uuid4 string), not INTEGER.
- `tasks.task_id` — **TEXT**.

Every foreign key in 002 (`first_seen_run_id`, `last_seen_run_id`,
`created_run_id`, `created_task_id`, `executed_run_id`) is TEXT to match. The
Sprint 2 plan had assumed INTEGER.

### Migration state is schema_migrations, not user_version

The plan assumed `PRAGMA user_version`. **It is not used anywhere.** The real
mechanism is the `schema_migrations` table — one row per applied filename,
created by 001 and bootstrapped by `migrate.py`. On a fully migrated database
`PRAGMA user_version` reads **0**, so anything trusting it would conclude no
migrations had ever run.

**CLAUDE.md had this wrong** and has been corrected — it claimed the memory
server would read `user_version` at startup and exit on mismatch. It now says
the memory server reads `schema_migrations` and requires 002 to be present and
applied. A test (`test_user_version_is_not_the_migration_marker`) pins this so
the two mechanisms cannot start disagreeing silently.

### llm_calls: role was an enforced CHECK, so it needed a rebuild

`role` is an **enforced CHECK constraint** from 001c, not a documented
convention:

```sql
role  TEXT NOT NULL
      CHECK (role IN ('planner', 'critic', 'writer', 'judge', 'router')),
```

`role='embedder'` was rejected with `IntegrityError: CHECK constraint failed`.
Since SQLite cannot widen a CHECK via `ALTER TABLE`, 002 rebuilds the table:
create new with `'embedder'` added, copy all 18 columns by name, drop, rename,
recreate both indexes, `PRAGMA foreign_key_check`. Nothing else about the
table changed.

**No `PRAGMA foreign_keys = OFF`, and 002 stays transactional.** The pragma is
in SQLite's 12-step procedure to protect the case where other tables reference
the one being dropped. Walking `PRAGMA foreign_key_list` over every table found
**zero referers of llm_calls** — it is a child (it points at `runs` and
`tasks`), never a parent. The full rebuild was verified to complete inside a
transaction with `foreign_keys=1` and `foreign_key_check` clean. Adding the
pragma would have forced 002 to become `-- migrate: no-transaction` like 001c,
costing the `facts`/`facts_vec`/`approvals` work its all-or-nothing guarantee
— and it would have been a silent no-op anyway, which is the exact trap 001c
exists to document.

**Column semantics for an embedding call.** Measured against a real
`nomic-embed-text` response, not assumed:

```
prompt_eval_count: 9          eval_count:    None
total_duration:    4302553600  eval_duration: None
load_duration:     2817178800  embeddings:    1 vector of len 768
```

- **Populated:** `call_id`, `run_id`, `role`, `model_tag`, `think_mode`,
  `prompt_eval_count`, `total_duration_ns`, `ok`, `timed_out`, `created_at`
  (and `error` on failure).
- **NULL:** `task_id` and `step_index` (the memory server is spawned per run
  and gets `--run-id` only; it is never told the current task), `prompt_file`
  (no template — the input is fact content), `eval_count`, `eval_duration_ns`,
  `tok_s`, and `raw_response` (the "response" is the 768-float vector, which
  lives in `facts_vec` rather than being duplicated here).

`eval_count`, `eval_duration_ns` and `tok_s` are **NULL, not 0**. Ollama's
embed response genuinely omits them — an embedding generates no output tokens
and has no generation phase to time. NULL means not applicable; 0 would assert
a measurement nobody took, and would silently drag any future `AVG(tok_s)`
toward zero.

`think_mode` is NOT NULL and thinking is not a concept for an embedding call,
so it records `'n/a'` rather than a misleading `'False'`.

### Tests

**79 passing, 0 failing** (`python -m unittest discover -s tests -v`), up from
52 at the end of Sprint 1. New in this step: the 002 forward migration and
clean re-run, the partial unique index firing on a second current row and
*not* firing on superseded rows, the `change_kind` and approvals `status`
CHECKs, duplicate `idempotency_key` rejection, `facts_vec` accepting 768
floats and rejecting 767, KNN over three vectors in distance order, the
empty-table KNN negative control, the `BEGIN IMMEDIATE` matched pair, two
concurrent writers landing 200 rows with zero `OperationalError`, and the
`llm_calls` rebuild preserving pre-existing rows and both indexes.

Also verified out-of-band, on a **copy** of the real `data/argus.db` (which
was at 001c): `002` applies as exactly one migration, a re-run is a clean
no-op, `integrity_check` returns ok, `foreign_key_check` is empty, and the
existing 10 runs / 70 checkpoints are preserved.

## Step 4: embedding module

`src/argus/embeddings.py` — async `embed()`, `serialize()`, and `llm_calls`
logging with `role='embedder'`, built against real `nomic-embed-text` calls.

### Empty string: neither error nor zeros

Negative control. `nomic-embed-text` returns an ordinary **unit vector**
(norm 1.0, non-zero components) for `""`, and `"   "` returns the
**byte-identical** vector to `""`. Not rejected, not zeroed — a real,
storable, queryable vector, indistinguishable downstream from a real fact.

`embeddings.py` does **not** reject empty input — silently rewriting a
caller's text is worse than embedding what was given. **The reject gate
belongs in `remember_fact` (Step 5)**, before the text ever reaches this
module.

### Prefix comparison — ship unprefixed

Measured against three fixed short strings (two similar, one unrelated):

| form | similar | unrelated | margin |
|---|---|---|---|
| no prefix | 0.4947 | 0.4738 | **+0.0209** |
| `search_document:` both sides | 0.5796 | 0.5939 | **−0.0142** (inverted) |
| `search_query:` / `search_document:` | 0.4589 | 0.4327 | +0.0262 |

Nomic's documented task prefixes did not help on this data: symmetric
`search_document:` prefixing **inverted** the ranking, and only the unprefixed
form gave a clean, correctly-signed margin among the two forms compared on
equal footing. **Decision: ship Step 6 with no prefix.** Revisit with a
larger measured set if Sprint 6/7 recall quality looks weak — three strings is
a smoke test, not a benchmark.

### The lru_cache-across-event-loops bug

`get_async_client` was first written as a plain `functools.lru_cache` keyed on
`(host, timeout)`. `httpx.AsyncClient` binds its connection pool to the event
loop that constructed it, so a client built under one loop and reused under a
different one fails with `RuntimeError: Event loop is closed`.

**Production would never have hit this** — the memory server runs a single
event loop for its entire life, so there is only ever one loop to key against.
**The test suite caught it** because `unittest.IsolatedAsyncioTestCase` opens
a fresh loop per test method; the second test to call `embed()` reused the
first test's now-dead loop's client and failed immediately.

Fixed by keying on the running loop as well, via a `weakref.WeakKeyDictionary`
so a dead loop's entry is collected with it. A regression test is committed
(`test_a_second_event_loop_does_not_inherit_a_dead_client`) that runs a second
loop in a real OS thread and asserts it gets a distinct client — this is the
one case an ordinary same-thread test cannot reproduce, since `asyncio.run()`
cannot nest inside a running loop.

### Layering fix: embeddings.py no longer imports argus.orchestrator.*

`argus.embeddings` originally imported `TIMEOUT_SECONDS`, `get_async_client`
and `log_call` from `argus.orchestrator.llm` — backwards, since the memory
server (a peer of the orchestrator, not a dependent) would have pulled in the
whole orchestrator package just to log an embedding call.

Extracted the shared plumbing (sync client cache, async client cache, `tok_s`,
`log_call`, `TIMEOUT_SECONDS`) into a new neutral module,
`src/argus/ollama_client.py`. `argus.orchestrator.llm` now imports from it and
re-exports the original names (`get_client as _get_client`, etc.) so nothing
in the orchestrator or its existing tests changed call signature or behavior.
`argus.embeddings` imports from `argus.ollama_client` directly.

Verified by a real import-graph check, not by eye:
`test_importing_embeddings_does_not_pull_in_orchestrator` runs
`import argus.embeddings` in a fresh subprocess interpreter and asserts
`'argus.orchestrator'` is absent from `sys.modules` afterward.

One necessary side effect: `test_llm.py` mocked the network seam via
`mock.patch.object(llm, "Client", RecordingClient)`. Python resolves a bare
name inside a function against *that function's own module's* globals, not the
importer's — so once `get_client`'s body moved to `ollama_client.py`, patching
`llm.Client` would silently stop intercepting anything (the function now reads
`ollama_client.Client`). The patch target was updated to
`mock.patch.object(ollama_client, "Client", RecordingClient)` so the mock keeps
doing what it always did — intercept the network call, not let it through.
This is the one test line that had to change to keep behavior, not signatures,
identical.

### Tests: 105 passing, 0 failing

Up from 104. New: the import-graph isolation check.

## Step 5: memory server skeleton + remember_fact

`src/argus/servers/memory.py`, first tool: `remember_fact`.

`RememberResult.change_kind`'s `"unchanged"` value is an **API-only outcome**
with no corresponding row write — it means the existing current row's
`last_seen_*` was touched and nothing else happened. `facts.change_kind`'s
CHECK constrains the column to `('added', 'modified', 'removed')`; `'unchanged'`
never appears there, by construction. **Sprint 4's diff logic needs to know
this asymmetry exists**: querying `facts.change_kind` will never surface an
`'unchanged'` row, because that outcome is defined as "no new row."
