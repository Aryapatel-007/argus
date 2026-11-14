-- Phase E: llm_calls learns about graph nodes; approvals learn about nodes
-- and human decisions.
--
-- llm_calls: role is an ENFORCED CHECK (001c, widened in 002). Graph nodes
-- log their node type as the role, so the CHECK is widened again (the
-- original six stay valid; a bogus role is still rejected). Five columns are
-- appended after the 18 locked by 001c, in this order:
--   graph_id, node_id, strategy, sample_index, decision_id
-- All nullable: legacy loop and embedder rows have no node.
--
-- Same rebuild pattern as 002 and still transactional: nothing references
-- llm_calls (it is only ever a child), so the DROP is legal with foreign key
-- enforcement on.

CREATE TABLE llm_calls_new (
    call_id            TEXT PRIMARY KEY,
    run_id             TEXT NOT NULL REFERENCES runs (run_id),
    task_id            TEXT REFERENCES tasks (task_id),
    step_index         INTEGER,
    role               TEXT NOT NULL
                       CHECK (role IN ('planner', 'critic', 'writer', 'judge',
                                       'router', 'embedder',
                                       'plan', 'replan', 'extract', 'critique',
                                       'resolve_entity', 'write_fact', 'summarize',
                                       'propose_action', 'reflect',
                                       'profiler', 'health')),
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
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    graph_id           TEXT,
    node_id            TEXT,
    strategy           TEXT,
    sample_index       INTEGER,
    decision_id        TEXT
);

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

CREATE INDEX IF NOT EXISTS idx_llm_calls_run ON llm_calls (run_id, created_at);
CREATE INDEX IF NOT EXISTS idx_llm_calls_task_step ON llm_calls (task_id, step_index);
CREATE INDEX IF NOT EXISTS idx_llm_calls_node ON llm_calls (node_id);
CREATE INDEX IF NOT EXISTS idx_llm_calls_model_time ON llm_calls (model_tag, created_at);

-- approvals: which graph node proposed it, and who decided.
ALTER TABLE approvals ADD COLUMN created_node_id TEXT;
ALTER TABLE approvals ADD COLUMN decided_by TEXT;
ALTER TABLE approvals ADD COLUMN decision_note TEXT;
