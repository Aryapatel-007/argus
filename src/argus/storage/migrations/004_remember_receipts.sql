-- Phase C, audit H3: remember_fact was not idempotent across a crash.
--
-- The memory server commits a fact on its own connection. If the caller dies
-- after that commit but before it checkpoints the result, the resumed caller
-- calls again and gets the wrong answer: 'unchanged' for a fact first seen in
-- this run, or a phantom 'modified' if the text differs.
--
-- A receipt is written in the SAME transaction as the fact write, keyed by a
-- caller-supplied idempotency key. A repeated call with the same key returns
-- the stored result byte for byte and writes nothing.
--
-- Owned by the memory server (like facts). Rows are tiny and never updated.

CREATE TABLE remember_receipts (
    idempotency_key  TEXT PRIMARY KEY,
    target_id        TEXT NOT NULL,
    entity_key       TEXT NOT NULL,
    -- RememberResult as JSON.
    result           TEXT NOT NULL,
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
