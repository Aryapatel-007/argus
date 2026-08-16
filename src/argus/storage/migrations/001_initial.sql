-- Sprint 1 initial schema.
-- Tables: schema_migrations, runs, tasks, checkpoints, llm_calls.
-- facts and approvals arrive in Sprint 2 as migration 002.

CREATE TABLE IF NOT EXISTS schema_migrations (
    filename    TEXT PRIMARY KEY,
    applied_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- One row per scheduled execution.
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    target_id    TEXT NOT NULL,
    status       TEXT NOT NULL,
    started_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    finished_at  TEXT,
    error        TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_target_started
    ON runs (target_id, started_at DESC);

-- Sub-tasks the planner generated within a run.
CREATE TABLE IF NOT EXISTS tasks (
    task_id      TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES runs (run_id),
    task_index   INTEGER NOT NULL,
    description  TEXT NOT NULL,
    tool         TEXT,
    status       TEXT NOT NULL,
    attempts     INTEGER NOT NULL DEFAULT 0,
    result       TEXT,
    error        TEXT,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_tasks_run
    ON tasks (run_id, task_index);

-- State after every transition. Written in the same transaction as the
-- transition itself, so the newest row is always the last completed step.
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id  TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL REFERENCES runs (run_id),
    task_id        TEXT REFERENCES tasks (task_id),
    step_index     INTEGER NOT NULL,
    state          TEXT NOT NULL,
    payload        TEXT,
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Resume reads the highest step_index for a run.
CREATE INDEX IF NOT EXISTS idx_checkpoints_run_step
    ON checkpoints (run_id, step_index DESC);

-- Every Ollama call. think_mode is recorded per call because think=False is
-- passed explicitly on every call and must be auditable after the fact.
CREATE TABLE IF NOT EXISTS llm_calls (
    call_id            TEXT PRIMARY KEY,
    run_id             TEXT NOT NULL REFERENCES runs (run_id),
    step_index         INTEGER,
    role               TEXT NOT NULL,
    model              TEXT NOT NULL,
    think_mode         TEXT NOT NULL,
    prompt_file        TEXT,
    prompt_eval_count  INTEGER,
    eval_count         INTEGER,
    total_duration_ns  INTEGER,
    eval_duration_ns   INTEGER,
    ok                 INTEGER NOT NULL,
    error              TEXT,
    raw_response       TEXT,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_run_step
    ON llm_calls (run_id, step_index);
