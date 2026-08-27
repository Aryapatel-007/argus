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

## Step 6: recall_facts

### Part A — the larger prefix probe (deferred from Step 4b)

Step 4b shipped "no prefix" on **three generic sentence pairs** and explicitly
flagged that as too small to certify a retrieval strategy. This is the promised
larger probe: **ten fact-shaped cases** — short realistic fragments of the kind
a browser-extraction tool actually produces (job postings, program
requirements, policy snippets), not general prose.

Each case is an (anchor, similar, unrelated) triple. Margin =
`cos(anchor, similar) − cos(anchor, unrelated)`. Positive means the model
ranked the genuinely related fragment above the unrelated one, which is the
only property retrieval needs. Negative is an **inversion** — the model
preferred the wrong fragment. Reproduce with `python scripts/probe_prefix.py`.

| case | none (sim/unrel/margin) | doc/doc | query/doc |
|---|---|---|---|
| job-title | 0.5906 / 0.6926 / **−0.1020** | 0.7293 / 0.7704 / **−0.0411** | 0.6133 / 0.6953 / **−0.0820** |
| salary | 0.6718 / 0.4929 / +0.1789 | 0.7387 / 0.5259 / +0.2127 | 0.6490 / 0.4630 / +0.1860 |
| intake-dates | 0.6334 / 0.4694 / +0.1639 | 0.7231 / 0.5156 / +0.2075 | 0.6647 / 0.5142 / +0.1505 |
| tuition | 0.8059 / 0.4420 / +0.3639 | 0.7949 / 0.4464 / +0.3484 | 0.7190 / 0.4430 / +0.2759 |
| entry-requirements | 0.7596 / 0.5013 / +0.2583 | 0.7621 / 0.5182 / +0.2439 | 0.7295 / 0.4821 / +0.2474 |
| visa-policy | 0.8347 / 0.5420 / +0.2927 | 0.8523 / 0.6417 / +0.2106 | 0.8057 / 0.5338 / +0.2719 |
| duration | 0.7303 / 0.4709 / +0.2594 | 0.7665 / 0.5519 / +0.2146 | 0.6874 / 0.4662 / +0.2211 |
| english-req | 0.7062 / 0.4128 / +0.2934 | 0.7573 / 0.5235 / +0.2337 | 0.6125 / 0.3919 / +0.2206 |
| location | 0.5489 / 0.5363 / +0.0126 | 0.6299 / 0.6632 / **−0.0332** | 0.5063 / 0.4787 / +0.0276 |
| skills | 0.6665 / 0.4219 / +0.2446 | 0.7575 / 0.5177 / +0.2398 | 0.6816 / 0.4619 / +0.2197 |
| **MEAN MARGIN** | **+0.1966** | +0.1837 | +0.1739 |
| **INVERSIONS** | **1/10** | 2/10 | **1/10** |
| **WORST MARGIN** | −0.1020 | −0.0411 | −0.0820 |

**Decision: ship unprefixed. The Step 4b call is CONFIRMED, not overturned.**
`none` has both the highest mean margin (+0.1966) and joint-fewest inversions
(1/10). Symmetric `doc/doc` prefixing is again the worst on inversions (2/10),
reproducing the direction of the 3-pair finding. There is no asymmetry to
implement: query text and stored fact text are embedded identically, with no
prefix on either side.

Two things the larger sample revealed that three pairs could not:

1. **Fact-shaped text separates ~10x better than generic prose.** The 3-pair
   probe's best margin was +0.0209; here it is +0.1966. The earlier number
   was pessimistic because generic sentences sit in a narrow cosine band. This
   materially raises confidence that KNN retrieval will work at all.
2. **`job-title` inverts under ALL THREE forms** (−0.1020 / −0.0411 /
   −0.0820), and `location` is near-zero. Both are cases where the anchor and
   the unrelated text are *structurally* the same kind of string — two job
   titles, two location statements — differing only in content. No prefix
   choice fixes this; it is a limitation of the embedding model on short
   structurally-parallel fragments. **Recorded as a known weakness, not
   solved here.** If Sprint 6/7 recall quality looks weak on short
   entity-like facts specifically, this table is the reason why, and the fix
   is a better model or hybrid lexical matching — not a prefix.

### KNN mechanics confirmed at sqlite-vec 0.1.9

Docs (`alexgarcia.xyz/sqlite-vec/features/knn.html`) state vec0's default
distance is **L2 (Euclidean)**, with `distance_metric=cosine` available at
table creation. Migration 002's `facts_vec` does **not** set it, so distances
come back as **L2**.

Measured, not assumed: **nomic-embed-text returns unit-normalized vectors** —
norms across four real embeddings were 0.9999999288, 0.9999998349,
1.0000000026, 1.0000000163. That matters because for unit vectors
`L2² = 2 − 2·cos`, so L2 and cosine rank **identically** and convert exactly.
Verified on real query/fact pairs, agreement to ~1e-7:

```
fact 4: vec0_L2=0.698116  cos=0.756316  sqrt(2-2cos)=0.698117  delta=3.79e-07
fact 3: vec0_L2=1.061570  cos=0.436535  sqrt(2-2cos)=1.061570  delta=1.41e-08
fact 2: vec0_L2=1.076214  cos=0.420882  sqrt(2-2cos)=1.076214  delta=8.90e-08
fact 1: vec0_L2=1.092918  cos=0.402765  sqrt(2-2cos)=1.092918  delta=1.01e-07
```

Also confirmed against the real extension:

- **`fact_id IN (...)` prefiltering works alongside `MATCH`** — so `target_id`
  narrowing happens PRE-KNN, not as a post-hoc trim of an already-limited
  result set. This is what lets `limit` be a true cap rather than an upper
  bound that unrelated-target rows silently eat into.
- `k = N` larger than the row count returns what exists (4 rows for k=100),
  no error.
- `ORDER BY distance LIMIT n` also works (SQLite 3.45.1, past the 3.41+ floor
  the docs give for LIMIT support); `k =` is used anyway as the documented
  portable form.
- KNN against an empty `facts_vec` returns zero rows and does not raise.

### Retrieval is phrasing-sensitive — a real limitation, found while testing

Writing the ranking test surfaced something the Part A table only hinted at.
Five fixture facts, all sharing a `last_seen_at` (so ordering is *pure*
relevance), queried five natural ways. Cosine against each fact:

| query | top-2 | correct? |
|---|---|---|
| `how much money is involved` | tuition 0.5579, **salary 0.4684** | **yes** |
| `what does it cost in dollars` | tuition 0.5253, intake 0.4455 | no |
| `what are the AUD amounts` | **intake 0.5891**, tuition 0.5237 | no |
| `annual fee and base pay amounts in AUD` | intake 0.6149, tuition 0.6123 | no |
| `when does it start and what does it cost` | tuition 0.5081, intake 0.4556 | no |

The sharpest case: **`"what are the AUD amounts"` ranks `"Intake dates:
February and July each year"` FIRST (0.5891)** — above the fact that literally
contains an AUD amount. `intake` contains no number, no currency, and nothing
about cost.

This is the same failure mode as Part A's `job-title` and `location`
inversions: nomic-embed-text is responding to *structural* similarity —
short, colon-delimited, factual-looking fragments — more than to semantic
content. Only 1 of 5 phrasings retrieved correctly.

**Not solved here, and the test was not weakened to hide it.**
`test_semantically_close_facts_rank_above_the_rest` uses the phrasing that
genuinely works and carries a comment pointing at this table. What this means
concretely:

- Relevance-only retrieval on short entity-like facts is **not reliable** with
  this embedding model. The 0.7/0.3 relevance/recency blend does not rescue
  it — recency is identical across these facts.
- The likely fixes are a stronger embedding model, or hybrid retrieval pairing
  vectors with lexical/keyword matching (BM25 or FTS5, already available in
  SQLite). Both are out of scope for Sprint 2.
- **Sprint 6/7 should treat this as the leading suspect** if recall quality
  looks weak, ahead of the scorer weights or the half-life — those are tunable
  knobs on a signal that is itself noisy here.

## Retrieval quality risk, carried forward

The ranking-test failure above raised a direct question: is the 4-of-5
phrasing failure a **prefix-form problem** (Part A picked a locally-optimal
form that a harder test would overturn), or something **upstream of prefix
choice** (the embedding model's discriminative power on short fact fragments,
which no prefix can fix)? This section answers that with a dedicated probe,
states the answer plainly, and records what it means for the rest of the
project.

### The original 5-phrasing table (from the ranking-test failure)

Five fixture facts sharing one `last_seen_at` (pure relevance, no recency
contribution), queried five natural ways, unprefixed:

| query | top-2 | correct? |
|---|---|---|
| `how much money is involved` | tuition 0.5579, **salary 0.4684** | **yes** |
| `what does it cost in dollars` | tuition 0.5253, intake 0.4455 | no |
| `what are the AUD amounts` | **intake 0.5891**, tuition 0.5237 | no |
| `annual fee and base pay amounts in AUD` | intake 0.6149, tuition 0.6123 | no |
| `when does it start and what does it cost` | tuition 0.5081, intake 0.4556 | no |

### The 5x3 table — same five phrasings, all three Part A prefix forms

| query | none | doc/doc | query/doc |
|---|---|---|---|
| what does it cost in dollars | tuition+intake — no | intake+tuition — no | tuition+intake — no |
| how much money is involved | tuition+salary — **OK** | intake+tuition — no | tuition+salary — **OK** |
| what are the AUD amounts | intake+tuition — no | intake+skills — no | intake+skills — no |
| annual fee and base pay amounts in AUD | intake+tuition — no | intake+tuition — no | salary+tuition — **OK** |
| when does it start and what does it cost | tuition+intake — no | intake+tuition — no | tuition+library — no |
| **CORRECT / 5** | **1/5** | **0/5** | **2/5** |

Full per-query cosine detail (reproduce with a probe identical in shape to
`scripts/probe_prefix.py`, run against these 5 fixture facts instead of the
Part A pairs):

```
'what does it cost in dollars'
  [      none] tuition=0.5253, intake=0.4455, library=0.3592, skills=0.3124, salary=0.2895   FAIL
  [   doc/doc] intake=0.6183, tuition=0.5578, skills=0.4937, library=0.4541, salary=0.4032   FAIL
  [ query/doc] tuition=0.5947, intake=0.4699, library=0.4427, salary=0.4307, skills=0.4061   FAIL

'how much money is involved'
  [      none] tuition=0.5579, salary=0.4684, intake=0.4251, skills=0.3862, library=0.3436   OK
  [   doc/doc] intake=0.6258, tuition=0.5914, skills=0.5789, salary=0.5665, library=0.4513   FAIL
  [ query/doc] tuition=0.5946, salary=0.5702, skills=0.4615, intake=0.4389, library=0.4166   OK

'what are the AUD amounts'
  [      none] intake=0.5891, tuition=0.5237, skills=0.4528, library=0.4299, salary=0.4173   FAIL
  [   doc/doc] intake=0.7300, skills=0.6265, salary=0.5212, tuition=0.5200, library=0.4872   FAIL
  [ query/doc] intake=0.5612, skills=0.5294, salary=0.5171, tuition=0.5077, library=0.4568   FAIL

'annual fee and base pay amounts in AUD'
  [      none] intake=0.6149, tuition=0.6123, salary=0.5697, skills=0.4926, library=0.4304   FAIL
  [   doc/doc] intake=0.6836, tuition=0.6127, salary=0.6025, skills=0.5812, library=0.4848   FAIL
  [ query/doc] salary=0.6307, tuition=0.6302, intake=0.5552, skills=0.5107, library=0.4701   OK

'when does it start and what does it cost'
  [      none] tuition=0.5081, intake=0.4556, library=0.4183, skills=0.3948, salary=0.3416   FAIL
  [   doc/doc] intake=0.6164, tuition=0.5518, skills=0.5351, library=0.5181, salary=0.4431   FAIL
  [ query/doc] tuition=0.5328, library=0.4912, intake=0.4461, skills=0.4315, salary=0.4091   FAIL
```

### Direct answer: upstream embedding-quality problem, not a prefix problem

**No form fixes most phrasings.** Best case (`query/doc`) still fails on 3 of
5. `none` (Part A's winner) gets 1/5, `doc/doc` gets 0/5. All three forms fail
on the majority of phrasings, and the best score (2/5 = 40%) is not "most" by
any reading. Per the criterion this probe was designed to distinguish: this is
**not** a prefix-tuning problem. The Part A "no prefix" decision is **not
overturned** — switching to `query/doc` on the strength of a 1-vs-2 gap at n=5
would be chasing noise, and the instruction not to prefix-tune further on this
finding is followed: no form selection change is made here.

The failure is upstream: nomic-embed-text's discriminative power on short,
structurally-parallel fact fragments at this length and style. This is the
same mechanism Part A's `job-title` and `location` inversions already showed
— restated here with a harder, multi-candidate ranking test rather than an
isolated pair comparison, and confirmed to persist across every prefix form
tried.

### Weak pattern search (n=5 — labeled weak, not a conclusion)

| query | words | overlap(tuition) | overlap(salary) | any form correct? |
|---|---|---|---|---|
| what does it cost in dollars | 6 | 0 | 0 | No |
| how much money is involved | 5 | 1 | 0 | **Yes** |
| what are the AUD amounts | 5 | 1 | 1 | No |
| annual fee and base pay amounts in AUD | 8 | 3 | 2 | **Yes** |
| when does it start and what does it cost | 9 | 0 | 0 | No |

No clean correlate. Query length does not predict success (8 words succeeds,
9 fails). Word overlap does not either: `"what are the AUD amounts"` has
literal token overlap with BOTH money facts (`AUD`) and still fails under
every form, while `"how much money is involved"` succeeds with almost no
lexical overlap at all — pure paraphrase. The only pattern visible at this
sample size is that the two queries with **zero word overlap and no
domain-specific noun** (`cost`, `involved` as the only content-bearing words)
split one success (paraphrase captured by the model) and one failure
(`"when does it start..."`, which mixes two questions into one embedding).
This is five data points; treat it as a place to look, not a finding.

### What this affects downstream

- **Sprint 4's entity-resolution / diffing.** If that logic calls
  `recall_facts` (or reimplements similar KNN candidate retrieval) to find the
  existing fact a new observation should compare against, the same
  discriminative-power limitation applies: a genuinely matching prior fact can
  rank below an unrelated one on short, structurally similar content.
- **Sprint 7's judge calibration.** Judge scoring assumes `recall_facts`
  surfaces the right candidate fact to evaluate. If retrieval itself is
  unreliable on short facts, a judge score is only as good as a candidate set
  that may not contain the right answer — a judge cannot rescue a retrieval
  miss.

### Candidate mitigations for later (named, not evaluated, not built)

1. **A larger or different embedding model.** Direct fix for discriminative
   power, at whatever VRAM/latency cost that carries — unmeasured.
2. **Hybrid keyword + vector search** — BM25 or SQLite's own FTS5, merged with
   KNN. Lexical matching would likely catch exactly the cases vectors miss
   here (e.g. the literal `AUD` overlap that cosine still ranked wrong).
3. **Query rewriting before embedding** — expanding or restructuring the
   query text ahead of the embed call. Unexplored; interacts with the prefix
   question rather than replacing it.

None of these are built in Sprint 2. This section exists so the next person
touching retrieval starts from a measured, characterized problem instead of
rediscovering it from a confusing ranking result.

## Step 7: list_changes_since

Third and final Sprint 2 tool. No facts_vec, no embed() call, no llm_calls
row — ever. Confirmed by test, not assumed: `TestListChangesNoEmbedding`
builds its fixture via direct SQL (bypassing `remember_fact` entirely) so the
guarantee holds independent of whether Ollama is even reachable, and asserts
`embed_call_count() == 0` after a real call through the tool. This makes
`list_changes_since` the one memory-server tool fully unaffected by the
retrieval-quality risk in Step 6 — if a fact's history looks wrong, the bug
is in `remember_fact` or upstream of it, never in embedding quality.

### created_at, not last_seen_at — the decision that looks equivalent until it isn't

"Changed since X" is answered against each row's own `created_at`, never
`last_seen_at`. The two columns move on different triggers:  `created_at` is
written once, at INSERT, and never touched again; `last_seen_at` is bumped on
every re-observation, including Step 5's `'unchanged'` path, which touches
`last_seen_at` on the *existing* row without writing a new one. A fact from
months ago that was simply re-observed yesterday would look like a fresh
change today under `last_seen_at` — silently wrong, and easy to not notice,
because both columns hold the same value on the common case (a fact seen once
and never touched again).

Proven under a real gap through the real write path, not just asserted in a
comment: `TestListChangesCreatedAtSemantics` writes an `'added'` fact, sleeps
0.2s, records a boundary timestamp, sleeps another 0.2s, writes a `'modified'`
version, then queries `since=boundary`. Result: only the `'modified'` row
comes back. Had the query used `last_seen_at` instead, the `'added'` row would
also have satisfied the filter — `last_seen_at` gets set to "now" on every
write, so both rows would pass regardless of which one actually changed
inside the window. The test would have passed by accident with the wrong
column; using a boundary strictly between the two real `created_at` values is
what makes the distinction observable at all.

### The previous_content contract for 'removed' rows — a requirement for Sprint 4, not yet a bug

`previous_content` is computed by ONE uniform rule for every `change_kind`:
a `LEFT JOIN facts prev ON prev.fact_id = f.supersedes_id`. There is no
`change_kind`-specific branch. This is deliberate — it means `'removed'`
support required zero removal-specific code, and `TestListChangesRemovedKind`
confirms it works correctly today, via a manually-inserted row (nothing in
the codebase produces `'removed'` yet; `remember_fact` only ever writes
`'added'` or `'modified'`).

But uniform-by-join means the correctness of `previous_content` for a
`'removed'` row is entirely contingent on that row actually setting
`supersedes_id` — the same contract `'modified'` rows already satisfy via
`remember_fact`. **This is a requirement Sprint 4 must implement deliberately
when it starts writing `'removed'` rows, not something to rediscover as a bug
after the fact:** if a future writer inserts a `'removed'` row without
`supersedes_id`, `list_changes_since` will not raise, will not warn — it will
silently return `previous_content: null` for a removal that clearly had prior
content. The failure is quiet by construction, which is exactly why it needs
to be a stated requirement here rather than left to be discovered.

### Tests: 171 passing, 0 failing

Up from 156. Fifteen new: the `created_at`-boundary proof, `previous_content`
correctness for `'added'`/`'modified'`, the `kinds` and `target_id` filters,
the manually-constructed `'removed'`-row pair, UTC-strictness for `since`
(unparseable, timezone-naive, and non-UTC-offset all rejected; `Z` and
`+00:00` both accepted as the same instant), the empty-window negative
control, `limit` as a true cap over more matches than the limit, and the two
no-embedding guarantees (`llm_calls` stays empty, `facts_vec` is never
touched).

## Step 9: Inspector session

No server behavior changed in this step — invocation and setup only.
`notes/inspector-session.md` is the runbook: the verified `npx
@modelcontextprotocol/inspector` stdio-launch syntax, the real dev database
(`data/argus.db`, migrated to 002 — see below), and a manual verification
checklist covering all three tools.

Fixing the invocation itself surfaced a repo-wide gap: `python -m
argus.servers.memory ...` run bare from the project root fails with
`ModuleNotFoundError: No module named 'argus'` — and the identical failure
reproduces on `runner.py`'s and `migrate.py`'s own documented `-m` commands.
This repo has no editable install (`pyproject.toml` / `pip install -e .`);
the only place the real convention (`PYTHONPATH=src`) is actually exercised
is `crash_test.py`'s subprocess spawning, invisibly. `memory.py`'s docstring
now documents both the `PYTHONPATH` form and the direct-script form
(`python src/argus/servers/memory.py ...`), matching the convention
`migrate.py` already had. `runner.py`'s docstring still lacks it — flagged,
not fixed, out of scope for this step.

`data/argus.db` was migrated to 002 to make a real Inspector session
possible: backed up first (`data/argus.db.pre-002-backup`, confirmed
byte-identical via SHA-256 before migrating), migration applied via the real
migration runner, `integrity_check` clean, zero FK violations, and
`runs`/`tasks`/`checkpoints`/`llm_calls` row counts matched exactly
before/after (10/10/70/25). Backup removed once the session closed out
successfully — no longer needed once 002 was confirmed stable in daily use.

### The run_id FK gap, surfaced by the first real session

The first real Inspector session — launched with `--run-id
inspector-session-1`, a string with no corresponding `runs` row — failed the
first `remember_fact` call with a bare `FOREIGN KEY constraint failed`,
naming neither table nor column. Diagnosis: not a type mismatch
(`runs.run_id` is TEXT, any string is valid) but a genuinely missing row.
Isolated which FK actually fires: **`llm_calls.run_id`, not `facts`** — the
embedding call's own logging insert runs before the facts write, so the
embed-logging path is what dies first.

**Decision: fail-fast startup check, not a loosened constraint.**
`facts.first_seen_run_id` / `last_seen_run_id` and `llm_calls.run_id` are
real provenance data — Sprint 4's diffing and Sprint 7's judge calibration
both depend on being able to answer "which run first saw this?" and "what
did this run cost?" A fact attributed to a run that never happened is
corrupt data, not a harmless label, so the FK stayed exactly as it was. The
invariant it enforces already holds by construction in real operation —
`loop._on_init` writes the `runs` row as the first database operation of any
run, before any tool call — so `assert_run_exists()` costs nothing in the
orchestrator-spawned path and only ever fires against an invented,
standalone `--run-id`. Full investigation, the isolation proof, and the
test suite are in Step 10 below.

### Confirmed: full manual Inspector session passes after the fix

Run against the real dev database, all three tools:

- Tool list showed exactly `remember_fact`, `recall_facts`,
  `list_changes_since` — no more, no fewer, schemas as documented.
- `remember_fact` correctly transitioned `added` → `unchanged` on a repeated
  identical call (not a duplicate `added`).
- `recall_facts` returned real `relevance` and `recency` components, not
  just a blended `score`.
- `list_changes_since` returned correct `previous_content` and `observed_at`
  semantics (created_at-based, not last_seen_at).
- `structured_content` was non-null on all three tool calls — the exact bug
  class fixed in Step 6, re-checked live rather than only in automated
  tests.
- Clean process exit; nothing printed to stdout beyond the MCP protocol
  frames Inspector itself renders.

Step 9 is closed. Cleanup of the throwaway run/fact created during this
manual session, and the pre-002 backup, is recorded at the end of Step 10.

## Step 10: the run_id FK gap — found by hand, missed by 171 tests

A real MCP Inspector session, launched with `--run-id inspector-session-1`,
failed on the first `remember_fact` call with a bare:

```
FOREIGN KEY constraint failed
```

Worth recording in full, because the interesting part is not the bug — it is
that **171 automated tests could not have caught it**, and thirty seconds of
manual driving did.

### What actually broke

Not a type mismatch. `runs.run_id` is `TEXT PRIMARY KEY` (real values look
like `run-10f35f5700b1`, from `loop.new_run`'s
`f"run-{uuid.uuid4().hex[:12]}"`), so `"inspector-session-1"` is a perfectly
valid TEXT value. It was purely a **missing row**.

Three foreign keys point at `runs.run_id` from this server's write path:

| table | column(s) |
|---|---|
| `llm_calls` | `run_id` |
| `facts` | `first_seen_run_id`, `last_seen_run_id` |
| `approvals` | `created_run_id`, `executed_run_id` |

Reproduced and isolated: **`llm_calls` fires first**, not `facts` — the
embedding call's `log_call` runs before the facts INSERT. Nothing partial
landed (`llm_calls=0, facts=0, facts_vec=0` after the failure), so the
transaction boundary held correctly; the only defect was the diagnosis.

### Why the test suite was blind to it

`MemoryServerTestCase.setUp` inserts a `runs` row before constructing the
server, and so does every other fixture in the suite — 13 `INSERT INTO runs`
call sites across 4 test files. **Every test constructs a world in which the
run already exists.** That is not sloppiness; it mirrors real operation. But
it means the suite never exercised the one thing standalone invocation does
differently: inventing a run id that was never started.

The general lesson, worth carrying into later sprints: a fixture that always
sets up the happy world will never test the setup step itself. Manual
driving found this precisely because Inspector does not run `setUp`.

### The design question, and the decision

Two real options, not just a patch choice:

**(a)** `--run-id` must reference a real `runs` row; validate at startup and
fail fast.
**(b)** `--run-id` is an opaque attribution label; drop or loosen the FKs so
`facts` tolerates a run id with no `runs` row.

**Chose (a).** Reasons, in order of weight:

1. **The architecture already guarantees it.** `loop._on_init` writes the
   `runs` row as the *first database operation of any run*, before PLAN,
   before ACT, before any tool call could happen. So in real
   orchestrator-spawned operation the invariant already holds by
   construction and the new check never fires. Option (a) costs nothing
   where it matters.
2. **The FKs are load-bearing, not decoration.** `facts.first_seen_run_id`
   is what makes "which run first saw this fact?" answerable — provenance
   that Sprint 4's diffing and Sprint 7's judge calibration both depend on.
   `llm_calls.run_id` is what makes per-run cost and latency accounting
   possible at all. A fact or an embedding cost attributed to a run that
   never happened is corrupt data, not a harmless label.
3. **(b) weakens a production invariant to serve a manual-testing
   convenience.** That trade runs the wrong way. And it is not even cheap:
   dropping an FK in SQLite requires a full table rebuild, i.e. a migration
   003, to support something one `INSERT` already solves.
4. **Standalone invocation was never first-class.** CLAUDE.md is explicit:
   "The memory server is spawned per run and receives `--run-id`". Nothing
   spawns it yet — Sprint 2 deliberately kept it standalone for testing —
   but spawned-per-run is the stated design, not an accident.

### What changed

The real defect was never the constraint; it was that the failure arrived
**late and cryptic**. `FOREIGN KEY constraint failed` names no table, no
column, and above all never mentions `--run-id`, which is the actual
mistake. So:

`assert_run_exists(db_path, run_id)` now runs in `main()`, immediately after
`assert_migrated()` and before anything is served. Order matters and is
tested: an unmigrated database has no `runs` table at all, so checking the
run first would raise `OperationalError` instead of the clear migration
message. The error names the offending id, explains why it cannot own facts,
and **lists real run ids from the database** — turning "that's wrong" into
"use this instead".

No schema change. No migration. No loosened constraint.

### Tests: 179 passing, 0 failing

Up from 171. Eight new: the missing-run rejection, the error naming valid
ids, the fresh-database "(none)" case, a control proving the assertion is
satisfiable (so the rejection tests are not passing merely because it always
raises), the real `run-{uuid4hex12}` shape, both `main()` end-to-end paths
(migration-check-first ordering, and bogus-run rejection through the real
entry point), and a no-regression test confirming that with a real run row
the full write path still succeeds and attributes `facts.first_seen_run_id`,
`facts.last_seen_run_id` and `llm_calls.run_id` to the correct run.

### Dev-database cleanup after Step 9's manual session

Two artifacts left behind by the manual investigation, both cleaned from
`data/argus.db` specifically — no test database or fixture was touched:

- `data/argus.db.pre-002-backup` — the pre-migration backup, confirmed
  byte-identical to the original via SHA-256 before migrating and no longer
  needed once 002 was confirmed stable. Deleted.
- The throwaway `run-8ff374e9cf6e` row and its dependents — one `facts` row
  (`fact_id 1`, `target_id inspector_test`), the matching `facts_vec` row,
  and 2 `llm_calls` rows (the embedder calls the manual session made).
  Deleted in FK-safe order inside one transaction: `facts_vec` → `facts` →
  `llm_calls` → `runs` (the run had to be last — `llm_calls.run_id` still
  referenced it up to that point). Confirmed no other table
  (`tasks`, `checkpoints`, `approvals`) referenced the throwaway run before
  deleting.

Row counts after cleanup: `runs=10, facts=0, facts_vec=0, llm_calls=25` —
an exact match against the state recorded before the Inspector session
began. `PRAGMA integrity_check` clean, zero FK violations. The real dev
database carries no debris from this investigation.
