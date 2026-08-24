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
