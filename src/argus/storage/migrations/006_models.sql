-- Phase E: what each model is good at, measured, and what the router learned.
--
-- config/models.yaml only says which models are ALLOWED. Everything about
-- what a model is good at lives here, written by scripts/profile_models.py
-- (priors) and by the bandit (posteriors). No tiers anywhere.

-- Key-value per (model, node type, metric). node_type '*' = model-wide
-- (vram_mb, cold_load_s, warm_load_s, tok_s, gpu_fraction, think_false_honored,
-- think_levels_coerced). Per node type: json_valid_rate, quality, stability,
-- mean_eval_tokens, mean_latency_s.
CREATE TABLE model_profiles (
    model_tag    TEXT NOT NULL,
    node_type    TEXT NOT NULL,
    metric       TEXT NOT NULL,
    value        REAL,
    samples      INTEGER NOT NULL DEFAULT 0,
    detail       TEXT,
    profile_run  TEXT,
    measured_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (model_tag, node_type, metric)
);

-- Pairs measured with `ollama ps`. model_a < model_b lexically.
CREATE TABLE co_residency (
    model_a      TEXT NOT NULL,
    model_b      TEXT NOT NULL,
    coresident   INTEGER NOT NULL CHECK (coresident IN (0, 1)),
    detail       TEXT,
    measured_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (model_a, model_b),
    CHECK (model_a < model_b)
);

-- Beta posterior per arm (node type, model). Seeded from the profile.
CREATE TABLE arm_stats (
    node_type   TEXT NOT NULL,
    model_tag   TEXT NOT NULL,
    alpha       REAL NOT NULL CHECK (alpha > 0),
    beta        REAL NOT NULL CHECK (beta > 0),
    pulls       INTEGER NOT NULL DEFAULT 0,
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (node_type, model_tag)
);

-- Every reward, once. UNIQUE makes applying a reward idempotent across
-- crash-resume: the arm_stats update happens in the same transaction and only
-- when this insert actually added a row.
CREATE TABLE arm_rewards (
    reward_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    node_type   TEXT NOT NULL,
    model_tag   TEXT NOT NULL,
    node_id     TEXT NOT NULL,
    source      TEXT NOT NULL CHECK (source IN ('validator', 'critic', 'eval', 'judge')),
    reward      REAL NOT NULL CHECK (reward >= 0 AND reward <= 1),
    detail      TEXT,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (node_id, model_tag, source)
);

CREATE TABLE model_health (
    model_tag             TEXT PRIMARY KEY,
    status                TEXT NOT NULL CHECK (status IN ('healthy', 'degraded')),
    reason                TEXT,
    recent_tok_s          REAL,
    baseline_tok_s        REAL,
    consecutive_timeouts  INTEGER NOT NULL DEFAULT 0,
    consecutive_slow      INTEGER NOT NULL DEFAULT 0,
    degraded_at           TEXT,
    updated_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
