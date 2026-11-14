-- Phase C, audit H2: nothing stopped two processes driving the same run.
--
-- Two guards, each sufficient on its own for a different failure:
--
--  1. UNIQUE (run_id, step_index) on checkpoints. If two writers ever race to
--     the same step, the second INSERT fails and its whole step transaction
--     rolls back. The database refuses the fork even if every application
--     check were bypassed.
--
--  2. run_leases: one owner per run. The owner's heartbeat is updated INSIDE
--     each step's transaction (loop.step), with a rowcount check, so a
--     process that has lost the lease cannot commit a single step. A lease is
--     taken over only when its owner process is dead or its heartbeat is
--     stale (storage/lease.py).
--
-- Transactional. If an existing database already holds duplicate
-- (run_id, step_index) rows the index creation fails and the whole migration
-- rolls back, leaving the data untouched.

CREATE UNIQUE INDEX IF NOT EXISTS idx_checkpoints_run_step_unique
    ON checkpoints (run_id, step_index);

CREATE TABLE run_leases (
    run_id        TEXT PRIMARY KEY REFERENCES runs (run_id),
    -- "<host>:<pid>:<random>". pid lets a resumer on the same host see that
    -- the previous owner is dead and take over at once instead of waiting
    -- for the heartbeat to go stale.
    owner         TEXT NOT NULL,
    acquired_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    heartbeat_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
