-- Locks the three task/checkpoint columns left undecided in 001.
-- Follow-up rather than an amendment to 001_initial.sql: 001 is already pushed
-- and already recorded in schema_migrations, so editing it in place would not
-- re-run and the file would stop describing the database.

-- Generated and stored BEFORE the ACT step runs, so a crash between "decided to
-- act" and "acted" is recoverable. Nullable: tasks that never reach ACT never
-- get one. Sprint 1 writes it; Sprint 5 adds the duplicate check that reads it.
ALTER TABLE tasks ADD COLUMN idempotency_key TEXT;

-- Semantics locked here, no schema change needed for either:
--
-- tasks.attempts  -- END-OF-TASK FINAL COUNT. Written once, when the task leaves
--                    the loop as DONE or FAILED. NOT incremented per retry.
--                    Live truth during a run is RunContext.attempt_count, which
--                    is checkpointed. Two tables must never both claim to be the
--                    live counter for the same value.
--
-- checkpoints.payload -- Full RunContext as JSON. Confirmed, keep as is.
