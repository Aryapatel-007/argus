-- migrate: no-transaction
-- Locks the schema decisions left open after 001/001b. Runs outside a
-- transaction (see marker above, documented in migrate.py) so that
-- PRAGMA foreign_keys = OFF actually takes effect for the tasks rebuild below.
-- Confirmed empirically: the same pragma issued mid-transaction reads back
-- unchanged and DROP TABLE tasks still fails FK enforcement while checkpoints
-- or llm_calls hold rows referencing it.
--
--   1. Removes tasks.idempotency_key (added by 001b, reversed here). The key
--      belongs to an external action, so it lands on `approvals` in Sprint 5.
--   2. tasks.attempts becomes NULLABLE: NULL = in flight or crashed, non-null =
--      terminated (structurally enforced by the CHECK below, not just by
--      convention). Live truth during a run is RunContext.attempt_count,
--      persisted via checkpoints only.
--   3. checkpoints.payload (full RunContext JSON) unchanged — confirmed.
--   4. llm_calls redefined with the full column set, see below.
--   5. tasks.status constrained to the task LIFECYCLE enum, not loop phases.
--      Loop phases (PLAN/ACT/OBSERVE/CRITIQUE/GATE/PERSIST) are a separate axis
--      and belong in state.py's State enum, not this column — do not unify them.

PRAGMA foreign_keys = OFF;

-- ---------------------------------------------------------------------------
-- tasks: rebuild (SQLite cannot drop NOT NULL or add CHECK via ALTER TABLE)
-- ---------------------------------------------------------------------------

CREATE TABLE tasks_new (
    task_id      TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES runs (run_id),
    task_index   INTEGER NOT NULL,
    description  TEXT NOT NULL,
    tool         TEXT,

    -- Lifecycle, not loop phase. Loop phases live in checkpoints.state.
    status       TEXT NOT NULL CHECK (status IN ('pending', 'running', 'done', 'failed')),

    -- Final count, written ONCE as the task leaves the loop. Never incremented
    -- per retry. NULL until termination.
    attempts     INTEGER,

    result       TEXT,
    error        TEXT,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),

    -- Makes "attempts is the terminal count" structural rather than a comment:
    -- terminated <=> attempts recorded.
    CHECK ((status IN ('done', 'failed')) = (attempts IS NOT NULL))
);

INSERT INTO tasks_new (
    task_id, run_id, task_index, description, tool, status, attempts,
    result, error, created_at
)
SELECT
    task_id, run_id, task_index, description, tool, status,
    CASE WHEN status IN ('done', 'failed') THEN attempts ELSE NULL END,
    result, error, created_at
FROM tasks;

DROP TABLE tasks;

ALTER TABLE tasks_new RENAME TO tasks;

CREATE INDEX IF NOT EXISTS idx_tasks_run ON tasks (run_id, task_index);

-- ---------------------------------------------------------------------------
-- llm_calls: redefined. Empty table, so dropped and recreated rather than
-- rebuilt. Every metric column is nullable so a call that times out or errors
-- still writes its row — the timeout is recorded, the insert is never skipped.
-- ---------------------------------------------------------------------------

DROP TABLE llm_calls;

CREATE TABLE llm_calls (
    call_id            TEXT PRIMARY KEY,
    run_id             TEXT NOT NULL REFERENCES runs (run_id),

    -- Nullable: planner calls happen before any task row exists.
    task_id            TEXT REFERENCES tasks (task_id),

    -- A task spans many steps (plan, act, observe, critique, retry,
    -- critique...). task_id says which task; step_index says which pass
    -- through the loop for that task. Sprint 6's trace tree needs both.
    step_index         INTEGER,

    role               TEXT NOT NULL
                       CHECK (role IN ('planner', 'critic', 'writer', 'judge', 'router')),
    model_tag          TEXT NOT NULL,

    -- CLAUDE.md requires think=False on every call, every role. Recorded per
    -- call so that is auditable after the fact rather than assumed.
    think_mode         TEXT NOT NULL,

    -- Which prompt template produced this call. Traces a behaviour change to a
    -- prompt edit instead of a guess.
    prompt_file        TEXT,

    prompt_eval_count  INTEGER,
    eval_count         INTEGER,
    -- Nanoseconds throughout, matching what Ollama returns natively — no ms
    -- conversion anywhere. eval_duration_ns kept alongside tok_s so the
    -- derived rate stays auditable against its inputs.
    eval_duration_ns   INTEGER,
    total_duration_ns  INTEGER,
    tok_s              REAL,

    ok                 INTEGER NOT NULL DEFAULT 0 CHECK (ok IN (0, 1)),
    timed_out          INTEGER NOT NULL DEFAULT 0 CHECK (timed_out IN (0, 1)),
    error              TEXT,

    -- The raw model output. What you want most at 2am debugging a
    -- schema-validation failure instead of guessing what the model said.
    raw_response       TEXT,

    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_run ON llm_calls (run_id, created_at);
CREATE INDEX IF NOT EXISTS idx_llm_calls_task_step ON llm_calls (task_id, step_index);

PRAGMA foreign_keys = ON;
