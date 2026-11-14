-- Phase E: graph engine state. Spec: notes/specs/graph-engine.md.
--
-- Owned by the orchestrator. Every node status change writes graph_nodes AND
-- a node_events row in ONE transaction (graph/store.py); node_events is the
-- checkpoint log, graph_nodes the current state that resume reads.

CREATE TABLE graphs (
    graph_id         TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL REFERENCES runs (run_id),
    target_id        TEXT NOT NULL,
    kind             TEXT NOT NULL CHECK (kind IN ('watch', 'reflect', 'legacy', 'eval')),
    status           TEXT NOT NULL
                     CHECK (status IN ('running', 'waiting_approval', 'done', 'failed')),
    -- Limits and run configuration (routing mode, budgets) as JSON, fixed at
    -- creation so a resumed graph runs under the rules it started with.
    limits           TEXT NOT NULL,
    config           TEXT NOT NULL DEFAULT '{}',
    budget_s         REAL,
    budget_tokens    INTEGER,
    expansions_used  INTEGER NOT NULL DEFAULT 0,
    tokens_used      INTEGER NOT NULL DEFAULT 0,
    -- Wall-clock seconds actually spent driving this graph, summed across
    -- resumes (a crash gap is not charged to the budget).
    seconds_used     REAL NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    finished_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_graphs_run ON graphs (run_id);
CREATE INDEX IF NOT EXISTS idx_graphs_status ON graphs (status, created_at);

CREATE TABLE graph_nodes (
    node_id            TEXT PRIMARY KEY,
    graph_id           TEXT NOT NULL REFERENCES graphs (graph_id),
    node_type          TEXT NOT NULL,
    status             TEXT NOT NULL
                       CHECK (status IN ('pending', 'ready', 'running', 'done',
                                         'failed', 'blocked', 'skipped')),
    params             TEXT NOT NULL DEFAULT '{}',
    attempt            INTEGER NOT NULL DEFAULT 1 CHECK (attempt >= 1),
    retry_of           TEXT REFERENCES graph_nodes (node_id),
    retry_root         TEXT NOT NULL,
    -- Side-effect nodes only. Stored with the node, i.e. before it can run.
    idempotency_key    TEXT UNIQUE,
    prunable           INTEGER NOT NULL CHECK (prunable IN (0, 1)),
    difficulty         REAL NOT NULL DEFAULT 0.5,
    template_instance  TEXT,
    expansion_id       TEXT,
    -- Written exactly once, in the same transaction as status -> done.
    output             TEXT,
    error              TEXT,
    skip_reason        TEXT,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    started_at         TEXT,
    finished_at        TEXT,
    -- A done node always has its output stored: downstream nodes read it
    -- from here, never from memory, so a resume cannot regenerate it.
    CHECK (status != 'done' OR output IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_graph_nodes_graph ON graph_nodes (graph_id, status);

CREATE TABLE graph_edges (
    graph_id  TEXT NOT NULL REFERENCES graphs (graph_id),
    src       TEXT NOT NULL REFERENCES graph_nodes (node_id),
    dst       TEXT NOT NULL REFERENCES graph_nodes (node_id),
    channel   TEXT NOT NULL,
    PRIMARY KEY (src, dst, channel)
);

CREATE INDEX IF NOT EXISTS idx_graph_edges_dst ON graph_edges (dst);

-- Every expansion attempt, accepted or not. Stored BEFORE its nodes can run.
CREATE TABLE graph_expansions (
    expansion_id  TEXT PRIMARY KEY,
    graph_id      TEXT NOT NULL REFERENCES graphs (graph_id),
    requested_by  TEXT,
    kind          TEXT NOT NULL CHECK (kind IN ('initial', 'plan', 'replan', 'retry', 'engine')),
    spec          TEXT NOT NULL,
    accepted      INTEGER NOT NULL CHECK (accepted IN (0, 1)),
    violations    TEXT,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- The checkpoint log. seq is per node, contiguous from 1.
CREATE TABLE node_events (
    event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    graph_id     TEXT NOT NULL REFERENCES graphs (graph_id),
    node_id      TEXT NOT NULL REFERENCES graph_nodes (node_id),
    seq          INTEGER NOT NULL,
    from_status  TEXT,
    to_status    TEXT NOT NULL,
    detail       TEXT,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (node_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_node_events_graph ON node_events (graph_id, event_id);

-- One row per completed model sample of a multi-call strategy. Stored before
-- the next sample starts; resume reuses them instead of regenerating.
CREATE TABLE node_samples (
    node_id       TEXT NOT NULL REFERENCES graph_nodes (node_id),
    sample_index  INTEGER NOT NULL,
    model_tag     TEXT NOT NULL,
    output        TEXT,
    ok            INTEGER NOT NULL CHECK (ok IN (0, 1)),
    error         TEXT,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (node_id, sample_index)
);

-- Why a model was chosen, answerable from SQL alone. `candidates` holds the
-- full score breakdown of EVERY candidate, not just the winner.
CREATE TABLE routing_decisions (
    decision_id   TEXT PRIMARY KEY,
    graph_id      TEXT NOT NULL REFERENCES graphs (graph_id),
    node_id       TEXT NOT NULL REFERENCES graph_nodes (node_id),
    node_type     TEXT NOT NULL,
    model_tag     TEXT NOT NULL,
    -- For cross_model / cascade: the ordered model list. JSON array.
    models        TEXT NOT NULL,
    strategy      TEXT NOT NULL
                  CHECK (strategy IN ('single', 'self_consistency', 'cascade',
                                      'cross_model', 'reflexion_retry')),
    candidates    TEXT NOT NULL,
    mode          TEXT NOT NULL CHECK (mode IN ('fixed', 'learning', 'baseline')),
    explore       INTEGER NOT NULL DEFAULT 0 CHECK (explore IN (0, 1)),
    -- Frozen in the same transaction as the node's ready -> running. A frozen
    -- decision is reused on resume, never recomputed.
    frozen        INTEGER NOT NULL DEFAULT 0 CHECK (frozen IN (0, 1)),
    superseded    INTEGER NOT NULL DEFAULT 0 CHECK (superseded IN (0, 1)),
    revised_from  TEXT REFERENCES routing_decisions (decision_id),
    reason        TEXT,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CHECK (NOT (frozen = 1 AND superseded = 1))
);

-- Exactly one live decision per node.
CREATE UNIQUE INDEX IF NOT EXISTS idx_routing_one_live
    ON routing_decisions (node_id) WHERE superseded = 0;

-- Procedural memory: which subgraph shapes work.
CREATE TABLE templates (
    template_id  TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    origin       TEXT NOT NULL CHECK (origin IN ('builtin', 'learned')),
    pattern      TEXT NOT NULL,
    uses         INTEGER NOT NULL DEFAULT 0,
    successes    INTEGER NOT NULL DEFAULT 0,
    promoted     INTEGER NOT NULL DEFAULT 0 CHECK (promoted IN (0, 1)),
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- One outcome row per template instance, so counting is idempotent on resume.
CREATE TABLE template_outcomes (
    template_instance  TEXT PRIMARY KEY,
    template_id        TEXT NOT NULL REFERENCES templates (template_id),
    graph_id           TEXT NOT NULL REFERENCES graphs (graph_id),
    success            INTEGER NOT NULL CHECK (success IN (0, 1)),
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
