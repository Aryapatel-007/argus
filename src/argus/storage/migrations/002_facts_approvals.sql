-- Sprint 2: facts (versioned knowledge), facts_vec (embeddings), approvals.
--
-- Transactional (no `-- migrate: no-transaction` marker): nothing here rebuilds
-- a table, so foreign key enforcement can stay on throughout.
--
-- NOT INCLUDED: `ALTER TABLE tasks DROP COLUMN idempotency_key`.
-- The Sprint 2 plan called for it, but the column is already gone: 001b added
-- it, and 001c reversed that in its tasks rebuild. Re-issuing the drop here
-- would fail with "no such column: idempotency_key" and block the migration.
-- The decision it encodes still stands and is now fully realised: the key lives
-- on `approvals` below, because an approval is queued in one run and executed in
-- a later one, and that gap is exactly the window the key protects. A task is
-- run-scoped and cannot span it. See design-decisions.md.

-- ---------------------------------------------------------------------------
-- facts: versioned knowledge per target
-- ---------------------------------------------------------------------------
--
-- Versioned, not updated in place. A change produces a NEW row with
-- version = previous + 1 and supersedes_id pointing back, and the old row flips
-- is_current to 0. History is therefore append-only and `list_changes_since` is
-- a query rather than a separate changelog table to keep in sync.

CREATE TABLE facts (
    fact_id            INTEGER PRIMARY KEY,

    target_id          TEXT    NOT NULL,

    -- Identity of the thing being tracked WITHIN a target, e.g. "intake_dates".
    -- Sprint 2 resolves entities by exact match on this column only; fuzzy
    -- matching is Sprint 4. Do not add normalisation here without that sprint.
    entity_key         TEXT    NOT NULL,

    content            TEXT    NOT NULL,

    -- Structured extraction as JSON, when the tool produced one. Nullable:
    -- plain-prose facts have no attributes.
    attributes         TEXT,

    -- Hash of the normalised content. The cheap "did this actually change?"
    -- check that runs before any embedding call, so an unchanged page costs
    -- zero Ollama time.
    content_hash       TEXT    NOT NULL,

    version            INTEGER NOT NULL DEFAULT 1,
    is_current         INTEGER NOT NULL DEFAULT 1 CHECK (is_current IN (0, 1)),

    -- Self-reference: the row this one replaced. NULL on a version 1 row.
    supersedes_id      INTEGER REFERENCES facts (fact_id),

    change_kind        TEXT    NOT NULL
                       CHECK (change_kind IN ('added', 'modified', 'removed')),

    confidence         REAL,

    -- Exists from Sprint 2 but stays NULL until Sprint 5. The retrieval scorer
    -- treats NULL as a neutral constant, so shipping the column early costs
    -- nothing and avoids a migration later. Do not start writing it here.
    importance         REAL,

    source_url         TEXT,

    -- TEXT to match runs.run_id, which is a TEXT uuid4 (001_initial.sql).
    first_seen_run_id  TEXT    NOT NULL REFERENCES runs (run_id),
    last_seen_run_id   TEXT    NOT NULL REFERENCES runs (run_id),

    first_seen_at      TEXT    NOT NULL,
    last_seen_at       TEXT    NOT NULL,
    created_at         TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- THE invariant: exactly one current row per (target_id, entity_key), enforced
-- by the database rather than by application code remembering to flip the old
-- row first. A partial index, so superseded rows (is_current = 0) are exempt and
-- history can hold as many versions as it likes.
CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_one_current
    ON facts (target_id, entity_key) WHERE is_current = 1;

-- Walking the version chain for one entity.
CREATE INDEX IF NOT EXISTS idx_facts_chain
    ON facts (target_id, entity_key, version);

-- "What did this run touch?" — drives list_changes_since.
CREATE INDEX IF NOT EXISTS idx_facts_lastseen
    ON facts (last_seen_run_id);

-- The unchanged-content short circuit, before embedding.
CREATE INDEX IF NOT EXISTS idx_facts_hash
    ON facts (target_id, content_hash);

-- ---------------------------------------------------------------------------
-- facts_vec: the embedding index
-- ---------------------------------------------------------------------------
--
-- 768 is fixed by nomic-embed-text (measured in notes/sprint-2.md, Step 0). It
-- is NOT a tunable — changing the embedding model means a new migration and a
-- full re-embed, not an edit here.
--
-- Plain form deliberately. vec0 0.1.9 does accept metadata columns, partition
-- keys and auxiliary columns (verified empirically against the installed
-- extension), but none of that is documented at
-- https://alexgarcia.xyz/sqlite-vec/ yet — the constructor and KNN sections are
-- literal TODO stubs. Filtering by target_id is done as a join against `facts`
-- instead. Revisit if and when the syntax is documented.
--
-- No foreign key to facts: SQLite virtual tables cannot carry one. fact_id
-- correspondence is maintained by the memory server, which writes both rows in
-- a single BEGIN IMMEDIATE transaction.

CREATE VIRTUAL TABLE facts_vec USING vec0(
    fact_id    INTEGER PRIMARY KEY,
    embedding  FLOAT[768]
);

-- ---------------------------------------------------------------------------
-- approvals: the human gate
-- ---------------------------------------------------------------------------
--
-- Hard constraint 4 in CLAUDE.md: anything that leaves the machine writes a row
-- here and blocks. Not a confidence threshold.

CREATE TABLE approvals (
    approval_id      INTEGER PRIMARY KEY,

    -- Generated and stored BEFORE the action runs. UNIQUE is the actual
    -- protection: a duplicate proposal cannot be inserted at all, so a crashed
    -- and resumed run cannot queue the same external action twice.
    idempotency_key  TEXT    NOT NULL UNIQUE,

    target_id        TEXT    NOT NULL,
    action_kind      TEXT    NOT NULL,

    -- The action as JSON, and the reasoning in prose. Both stored so a human
    -- approving this in a week can see what it does and why it was proposed.
    proposed_action  TEXT    NOT NULL,
    reasoning        TEXT    NOT NULL,

    confidence       REAL,

    -- 'approved' and 'executed' are deliberately SEPARATE. The gap between them
    -- is the crash window idempotency_key exists for: approved-but-not-yet-
    -- executed is a real, recoverable state. Do not collapse them.
    status           TEXT    NOT NULL
                     CHECK (status IN
                       ('pending', 'approved', 'rejected', 'executed', 'failed')),

    -- TEXT to match runs.run_id / tasks.task_id.
    created_run_id   TEXT    NOT NULL REFERENCES runs (run_id),
    created_task_id  TEXT    REFERENCES tasks (task_id),

    -- Nullable and usually a DIFFERENT run from created_run_id — that is the
    -- whole point of the table.
    executed_run_id  TEXT    REFERENCES runs (run_id),

    created_at       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    resolved_at      TEXT,
    executed_at      TEXT,
    result           TEXT
);

-- The queue read: pending items, oldest first.
CREATE INDEX IF NOT EXISTS idx_approvals_pending
    ON approvals (status, created_at);

-- ---------------------------------------------------------------------------
-- llm_calls: widen the role CHECK to admit 'embedder'
-- ---------------------------------------------------------------------------
--
-- role is an ENFORCED CHECK constraint (001c), not a documented convention, so
-- the memory server's embedding calls cannot be logged without this. Verified:
-- inserting role='embedder' against the 001c schema fails with
-- "CHECK constraint failed: role IN ('planner','critic','writer','judge','router')".
--
-- SQLite cannot widen a CHECK via ALTER TABLE, so this is the standard
-- rebuild: create new, copy, drop, rename, recreate indexes, verify.
--
-- NO `PRAGMA foreign_keys = OFF` here, deliberately, and this migration stays
-- TRANSACTIONAL as a result. The pragma is in SQLite's published 12-step
-- procedure to protect the case where OTHER tables reference the table being
-- dropped. Nothing references llm_calls — it is a child (it points at runs and
-- tasks), never a parent. Confirmed by walking PRAGMA foreign_key_list over
-- every table: zero referers. So the DROP is legal with enforcement left on,
-- and the whole rebuild was verified to complete inside a transaction with
-- foreign_keys=1 and PRAGMA foreign_key_check clean.
--
-- This matters more than it looks. Adding the pragma would have forced this
-- file to become `-- migrate: no-transaction` like 001c, which would have cost
-- the facts/facts_vec/approvals work above its all-or-nothing guarantee — and
-- the pragma would have been a SILENT NO-OP anyway, exactly the trap 001c
-- documents. A line that does nothing is worse than no line.
--
-- All 18 columns preserved exactly, in order. Nothing else about this table is
-- changed: same types, same defaults, same CHECKs on ok/timed_out, same FKs.
--
-- COLUMN SEMANTICS FOR AN EMBEDDING CALL (role='embedder'):
-- Measured against a real nomic-embed-text response, not assumed.
--
--   POPULATED: call_id, run_id, role, model_tag, think_mode, total_duration_ns,
--              prompt_eval_count, ok, timed_out, created_at
--              (error too, on failure)
--
--   NULL:      task_id, step_index  -- the memory server is spawned per RUN and
--                                      receives --run-id only; it is never told
--                                      the orchestrator's current task.
--              prompt_file          -- no template; the input is fact content.
--              eval_count           -- an embedding generates NO output tokens.
--              eval_duration_ns     -- there is no generation phase to time.
--              tok_s                -- derived from the two above.
--              raw_response         -- the "response" is the 768-float vector;
--                                      it lives in facts_vec, not duplicated here.
--
-- eval_count, eval_duration_ns and tok_s are NULL, NOT 0. Ollama's embed
-- response genuinely omits them (verified: eval_count=None, eval_duration=None,
-- prompt_eval_count=9, total_duration=4302553600). NULL means not applicable;
-- 0 would assert a measurement that was never taken, and would silently drag
-- any future AVG(tok_s) toward zero.
--
-- think_mode is NOT NULL and thinking is not a concept for an embedding call,
-- so it records the string 'n/a' rather than a misleading 'False'.

CREATE TABLE llm_calls_new (
    call_id            TEXT PRIMARY KEY,
    run_id             TEXT NOT NULL REFERENCES runs (run_id),
    task_id            TEXT REFERENCES tasks (task_id),
    step_index         INTEGER,

    -- 'embedder' added. The memory server logs its own embedding calls, so
    -- llm_calls stays the single record of every Ollama call in the system even
    -- though a second process now makes some of them.
    role               TEXT NOT NULL
                       CHECK (role IN ('planner', 'critic', 'writer', 'judge',
                                       'router', 'embedder')),
    model_tag          TEXT NOT NULL,
    think_mode         TEXT NOT NULL,
    prompt_file        TEXT,
    prompt_eval_count  INTEGER,
    eval_count         INTEGER,
    eval_duration_ns   INTEGER,
    total_duration_ns  INTEGER,
    tok_s              REAL,
    ok                 INTEGER NOT NULL DEFAULT 0 CHECK (ok IN (0, 1)),
    timed_out          INTEGER NOT NULL DEFAULT 0 CHECK (timed_out IN (0, 1)),
    error              TEXT,
    raw_response       TEXT,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Columns named explicitly rather than SELECT *, so a future column-order
-- change cannot silently shift data one column to the left.
INSERT INTO llm_calls_new (
    call_id, run_id, task_id, step_index, role, model_tag, think_mode,
    prompt_file, prompt_eval_count, eval_count, eval_duration_ns,
    total_duration_ns, tok_s, ok, timed_out, error, raw_response, created_at
)
SELECT
    call_id, run_id, task_id, step_index, role, model_tag, think_mode,
    prompt_file, prompt_eval_count, eval_count, eval_duration_ns,
    total_duration_ns, tok_s, ok, timed_out, error, raw_response, created_at
FROM llm_calls;

DROP TABLE llm_calls;

ALTER TABLE llm_calls_new RENAME TO llm_calls;

-- Both indexes the original carried, recreated verbatim. (The implicit
-- sqlite_autoindex on the TEXT PRIMARY KEY comes back on its own.)
CREATE INDEX IF NOT EXISTS idx_llm_calls_run ON llm_calls (run_id, created_at);
CREATE INDEX IF NOT EXISTS idx_llm_calls_task_step ON llm_calls (task_id, step_index);

-- Belt and braces: a rebuild that silently orphaned a row would show up here.
PRAGMA foreign_key_check;
