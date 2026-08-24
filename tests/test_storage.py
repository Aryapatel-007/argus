"""Storage layer tests.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import random
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sqlite_vec  # noqa: E402

from argus.storage import db  # noqa: E402
from argus.storage.migrate import MIGRATIONS_DIR, apply_migrations  # noqa: E402

EXPECTED_TABLES = {
    "schema_migrations", "runs", "tasks", "checkpoints", "llm_calls",
    # 002
    "facts", "facts_vec", "approvals",
}
EXPECTED_MIGRATIONS = [
    "001_initial.sql", "001b_task_columns.sql", "001c_lock_schema.sql",
    "002_facts_approvals.sql",
]

#: Fixed by nomic-embed-text. Not a tunable — see 002_facts_approvals.sql.
EMBEDDING_DIMS = 768

# Locked by 001c: call_id..created_at, in this order. task_id/step_index both
# kept (which task, which pass through the loop); prompt_file and raw_response
# restored for traceability and 2am debugging.
EXPECTED_LLM_CALLS_COLUMNS = [
    "call_id", "run_id", "task_id", "step_index", "role", "model_tag",
    "think_mode", "prompt_file", "prompt_eval_count", "eval_count",
    "eval_duration_ns", "total_duration_ns", "tok_s", "ok", "timed_out",
    "error", "raw_response", "created_at",
]


def task_columns(db_path: Path) -> list:
    with db.connect(db_path) as conn:
        return [row["name"] for row in conn.execute("PRAGMA table_info(tasks)")]


def llm_calls_columns(db_path: Path) -> list:
    with db.connect(db_path) as conn:
        return [row["name"] for row in conn.execute("PRAGMA table_info(llm_calls)")]


def apply_only(db_path: Path, filename: str) -> None:
    """Build a database that has run exactly one migration.

    Deliberately does not reuse migrate.py's statement splitter, so the upgrade
    test's starting state is independent of the code under test.
    """
    sql = (MIGRATIONS_DIR / filename).read_text(encoding="utf-8")
    with db.connect(db_path) as conn:
        conn.executescript(sql)
        conn.execute("INSERT INTO schema_migrations (filename) VALUES (?)", (filename,))


class Boom(Exception):
    """Stand-in for a crash partway through a state transition."""


def table_names(db_path: Path) -> set:
    with db.connect(db_path) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        return {row["name"] for row in rows}


def count(db_path: Path, table: str) -> int:
    """Row count read on a FRESH connection, so only committed data is visible."""
    with db.connect(db_path) as conn:
        return conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]


def random_vector(dims: int = EMBEDDING_DIMS) -> bytes:
    """A serialized random embedding.

    Deliberately random, not from a real model: the embedding module is the
    next step, and nothing here tests semantic quality — only that the column
    width, the round trip and the KNN mechanics work.
    """
    return sqlite_vec.serialize_float32([random.random() for _ in range(dims)])


def insert_run(conn, run_id: str = "run-1") -> str:
    conn.execute(
        "INSERT INTO runs (run_id, target_id, status) VALUES (?, ?, ?)",
        (run_id, "monash_ai_masters", "running"),
    )
    return run_id


def insert_fact(conn, fact_id: int, entity_key: str, **overrides) -> None:
    """One facts row with every NOT NULL column filled and sane defaults."""
    row = {
        "fact_id": fact_id,
        "target_id": "monash_ai_masters",
        "entity_key": entity_key,
        "content": "Intake dates: February and July.",
        "content_hash": f"hash-{fact_id}",
        "is_current": 1,
        "change_kind": "added",
        "first_seen_run_id": "run-1",
        "last_seen_run_id": "run-1",
        "first_seen_at": "2026-08-24T00:00:00.000Z",
        "last_seen_at": "2026-08-24T00:00:00.000Z",
    }
    row.update(overrides)
    columns = ", ".join(row)
    placeholders = ", ".join("?" for _ in row)
    conn.execute(f"INSERT INTO facts ({columns}) VALUES ({placeholders})", tuple(row.values()))


def insert_approval(conn, approval_id: int, idempotency_key: str, **overrides) -> None:
    row = {
        "approval_id": approval_id,
        "idempotency_key": idempotency_key,
        "target_id": "monash_ai_masters",
        "action_kind": "send_email",
        "proposed_action": '{"to": "me@example.com"}',
        "reasoning": "Intake dates changed.",
        "status": "pending",
        "created_run_id": "run-1",
    }
    row.update(overrides)
    columns = ", ".join(row)
    placeholders = ", ".join("?" for _ in row)
    conn.execute(
        f"INSERT INTO approvals ({columns}) VALUES ({placeholders})", tuple(row.values())
    )


def insert_run_task_checkpoint(conn) -> None:
    """The Sprint 1 unit of work: a transition plus its checkpoint."""
    conn.execute(
        "INSERT INTO runs (run_id, target_id, status) VALUES (?, ?, ?)",
        ("run-1", "monash_ai_masters", "running"),
    )
    conn.execute(
        "INSERT INTO tasks (task_id, run_id, task_index, description, status) "
        "VALUES (?, ?, ?, ?, ?)",
        ("task-1", "run-1", 0, "check intake dates", "pending"),
    )
    conn.execute(
        "INSERT INTO checkpoints (checkpoint_id, run_id, task_id, step_index, state) "
        "VALUES (?, ?, ?, ?, ?)",
        ("ckpt-1", "run-1", "task-1", 0, "plan"),
    )


class StorageTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "test_argus.db"
        self.addCleanup(self._tmp.cleanup)


class TestMigrations(StorageTestCase):
    def test_fresh_db_gets_all_expected_tables(self) -> None:
        applied = apply_migrations(self.db_path)

        self.assertEqual(applied, EXPECTED_MIGRATIONS)
        self.assertTrue(EXPECTED_TABLES.issubset(table_names(self.db_path)))

    def test_migrations_apply_in_filename_order(self) -> None:
        """001b must sort after 001 and 001c before 002."""
        found = sorted(p.name for p in MIGRATIONS_DIR.glob("*.sql"))
        self.assertEqual(found, EXPECTED_MIGRATIONS)

    def test_wal_mode_is_on(self) -> None:
        apply_migrations(self.db_path)
        with db.connect(self.db_path) as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(mode.lower(), "wal")

    def test_rerunning_applies_nothing(self) -> None:
        """Second run of the full chain is a clean no-op.

        Asserted against schema_migrations, which is the REAL migration-state
        mechanism here. PRAGMA user_version is never written by this codebase —
        it reads 0 on a fully migrated database, so anything that trusts it
        would conclude no migrations had run at all.
        """
        apply_migrations(self.db_path)
        self.assertEqual(apply_migrations(self.db_path), [])

        recorded = [r["filename"] for r in self._migration_rows()]
        self.assertEqual(recorded, EXPECTED_MIGRATIONS)
        self.assertEqual(count(self.db_path, "schema_migrations"), len(EXPECTED_MIGRATIONS))

    def test_user_version_is_not_the_migration_marker(self) -> None:
        """Guards the assumption above rather than leaving it in a comment.

        If a later sprint starts writing PRAGMA user_version, this fails and
        forces a decision about which mechanism is authoritative — instead of
        the two silently disagreeing.
        """
        apply_migrations(self.db_path)
        with db.connect(self.db_path) as conn:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 0)

    def _migration_rows(self):
        with db.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT filename FROM schema_migrations ORDER BY filename"
            ).fetchall()


class TestPost001cSchema(StorageTestCase):
    """Schema decisions locked by 001c.

    idempotency_key was added by 001b and reversed here — it belongs on
    `approvals` in Sprint 5, not on tasks. attempts is now nullable: NULL means
    in flight or crashed, non-null means terminated, enforced structurally by
    a CHECK rather than left as a convention. llm_calls carries its final
    18-column set.
    """

    def test_fresh_db_has_no_idempotency_key(self) -> None:
        apply_migrations(self.db_path)
        self.assertNotIn("idempotency_key", task_columns(self.db_path))

    def test_llm_calls_has_the_locked_18_columns(self) -> None:
        apply_migrations(self.db_path)
        self.assertEqual(llm_calls_columns(self.db_path), EXPECTED_LLM_CALLS_COLUMNS)

    def test_upgrade_from_001b_drops_key_and_reinterprets_attempts(self) -> None:
        """001b's database (idempotency_key present, attempts NOT NULL DEFAULT
        0) must come out the other side of 001c with the key gone and attempts
        correctly reinterpreted by status — not just carried over raw. A
        pending task under the OLD schema always had attempts=0 (the column's
        default); the CHECK's meaning only holds if the migration turns that
        into NULL for anything not terminal.
        """
        apply_only(self.db_path, "001_initial.sql")
        apply_only(self.db_path, "001b_task_columns.sql")

        with db.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO runs (run_id, target_id, status) VALUES (?, ?, ?)",
                ("run-1", "monash_ai_masters", "running"),
            )
            conn.execute(
                "INSERT INTO tasks "
                "(task_id, run_id, task_index, description, status, attempts, "
                "idempotency_key) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("task-done", "run-1", 0, "check intake dates", "done", 3,
                 "run-1:task-done:act:0"),
            )
            conn.execute(
                "INSERT INTO tasks "
                "(task_id, run_id, task_index, description, status, attempts, "
                "idempotency_key) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("task-pending", "run-1", 1, "check fees", "pending", 0, None),
            )

        self.assertEqual(
            apply_migrations(self.db_path),
            ["001c_lock_schema.sql", "002_facts_approvals.sql"],
        )
        self.assertNotIn("idempotency_key", task_columns(self.db_path))

        with db.connect(self.db_path) as conn:
            done_row = conn.execute(
                "SELECT status, attempts FROM tasks WHERE task_id = ?", ("task-done",)
            ).fetchone()
            pending_row = conn.execute(
                "SELECT status, attempts FROM tasks WHERE task_id = ?", ("task-pending",)
            ).fetchone()

        self.assertEqual(done_row["status"], "done")
        self.assertEqual(done_row["attempts"], 3)
        self.assertEqual(pending_row["status"], "pending")
        self.assertIsNone(pending_row["attempts"])


class TestCheckConstraints(StorageTestCase):
    """The two 001c CHECK constraints: they must fail loudly, not silently
    accept a bad row. Each test asserts sqlite3.IntegrityError specifically,
    not just "some exception", so a constraint that's silently dropped in a
    future rebuild is caught rather than passing on any raised error.
    """

    def setUp(self) -> None:
        super().setUp()
        apply_migrations(self.db_path)
        with db.transaction(self.db_path) as conn:
            conn.execute(
                "INSERT INTO runs (run_id, target_id, status) VALUES (?, ?, ?)",
                ("run-1", "monash_ai_masters", "running"),
            )

    def test_done_task_with_null_attempts_is_rejected(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            with db.transaction(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO tasks "
                    "(task_id, run_id, task_index, description, status, attempts) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    ("task-1", "run-1", 0, "check intake dates", "done", None),
                )

    def test_pending_task_with_non_null_attempts_is_rejected(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            with db.transaction(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO tasks "
                    "(task_id, run_id, task_index, description, status, attempts) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    ("task-1", "run-1", 0, "check intake dates", "pending", 1),
                )

    def test_invalid_task_status_is_rejected(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            with db.transaction(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO tasks "
                    "(task_id, run_id, task_index, description, status) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("task-1", "run-1", 0, "check intake dates", "not_a_real_status"),
                )

    def test_invalid_llm_calls_role_is_rejected(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            with db.transaction(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO llm_calls (call_id, run_id, role, model_tag, think_mode) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("call-1", "run-1", "not_a_real_role", "qwen3.5:9b", "False"),
                )


class TestTransactions(StorageTestCase):
    def setUp(self) -> None:
        super().setUp()
        apply_migrations(self.db_path)

    def test_run_task_checkpoint_commit_together(self) -> None:
        with db.transaction(self.db_path) as conn:
            insert_run_task_checkpoint(conn)

        self.assertEqual(count(self.db_path, "runs"), 1)
        self.assertEqual(count(self.db_path, "tasks"), 1)
        self.assertEqual(count(self.db_path, "checkpoints"), 1)

    def test_failure_mid_transaction_writes_nothing(self) -> None:
        """The crash-recovery guarantee: a partial step leaves no trace.

        Everything before the raise is already executed on the connection. If
        the rollback did not hold, runs and tasks would survive without their
        checkpoint and a resume would replay from a state that never completed.
        """
        with self.assertRaises(Boom):
            with db.transaction(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO runs (run_id, target_id, status) VALUES (?, ?, ?)",
                    ("run-crash", "monash_ai_masters", "running"),
                )
                conn.execute(
                    "INSERT INTO tasks (task_id, run_id, task_index, description, status) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("task-crash", "run-crash", 0, "check intake dates", "pending"),
                )
                raise Boom("crash after the task insert, before the checkpoint")

        self.assertEqual(count(self.db_path, "runs"), 0)
        self.assertEqual(count(self.db_path, "tasks"), 0)
        self.assertEqual(count(self.db_path, "checkpoints"), 0)

    def test_failure_does_not_discard_earlier_committed_work(self) -> None:
        """A rolled-back step must not take the previous committed step with it."""
        with db.transaction(self.db_path) as conn:
            insert_run_task_checkpoint(conn)

        with self.assertRaises(Boom):
            with db.transaction(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO checkpoints "
                    "(checkpoint_id, run_id, task_id, step_index, state) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("ckpt-2", "run-1", "task-1", 1, "act"),
                )
                raise Boom("crash during the second step")

        self.assertEqual(count(self.db_path, "runs"), 1)
        self.assertEqual(count(self.db_path, "checkpoints"), 1)
        with db.connect(self.db_path) as conn:
            latest = conn.execute(
                "SELECT step_index, state FROM checkpoints "
                "WHERE run_id = ? ORDER BY step_index DESC LIMIT 1",
                ("run-1",),
            ).fetchone()
        self.assertEqual(latest["step_index"], 0)
        self.assertEqual(latest["state"], "plan")

    def test_broken_atomic_negative_control_lets_partial_writes_survive(self) -> None:
        """Negative control for test_failure_mid_transaction_writes_nothing.

        That test is the assertion Sprint 1's crash-recovery guarantee rests
        on. This proves it isn't vacuous: swap in an `atomic()` that COMMITs on
        exception instead of rolling back (deliberately broken, local to this
        test, never exported), run the identical insert-then-raise sequence,
        and confirm the rows DO survive. If `db.atomic()` were ever broken this
        way for real, the real test would start passing for the wrong reason -
        this test is what would notice.
        """

        @contextmanager
        def broken_atomic(conn: sqlite3.Connection):
            conn.execute("BEGIN")
            try:
                yield conn
            except BaseException:
                conn.execute("COMMIT")  # deliberately wrong: should ROLLBACK
                raise
            else:
                conn.execute("COMMIT")

        with self.assertRaises(Boom):
            with db.connect(self.db_path) as conn:
                with broken_atomic(conn):
                    conn.execute(
                        "INSERT INTO runs (run_id, target_id, status) VALUES (?, ?, ?)",
                        ("run-crash", "monash_ai_masters", "running"),
                    )
                    conn.execute(
                        "INSERT INTO tasks (task_id, run_id, task_index, description, status) "
                        "VALUES (?, ?, ?, ?, ?)",
                        ("task-crash", "run-crash", 0, "check intake dates", "pending"),
                    )
                    raise Boom("crash after the task insert, before the checkpoint")

        self.assertEqual(count(self.db_path, "runs"), 1)
        self.assertEqual(count(self.db_path, "tasks"), 1)
        self.assertEqual(count(self.db_path, "checkpoints"), 0)


class TestNoTransactionMarker(StorageTestCase):
    """Proves the `-- migrate: no-transaction` mechanism in migrate.py itself.

    Uses a minimal parent/child fixture unrelated to the real schema, so this
    tests the marker mechanism directly rather than 001c's side effects.
    Mirrors 001c's actual problem: DROP TABLE parent while a referencing child
    row exists. `PRAGMA foreign_keys = OFF` only makes that succeed when issued
    outside a transaction.
    """

    def _write_fixture(self, rebuild_sql: str) -> Path:
        migrations_dir = Path(self._tmp.name) / "fixture_migrations"
        migrations_dir.mkdir()
        (migrations_dir / "000_setup.sql").write_text(
            "CREATE TABLE parent (id INTEGER PRIMARY KEY);\n"
            "CREATE TABLE child (id INTEGER PRIMARY KEY, "
            "parent_id INTEGER REFERENCES parent (id));\n"
            "INSERT INTO parent VALUES (1);\n"
            "INSERT INTO child VALUES (1, 1);\n",
            encoding="utf-8",
        )
        (migrations_dir / "001_rebuild.sql").write_text(rebuild_sql, encoding="utf-8")
        return migrations_dir

    def test_transactional_migration_cannot_disable_foreign_keys(self) -> None:
        """Documents the bug 001c exists to avoid: PRAGMA OFF mid-transaction
        is a silent no-op, so DROP TABLE parent still fails FK enforcement.
        """
        migrations_dir = self._write_fixture(
            "PRAGMA foreign_keys = OFF;\n"
            "DROP TABLE parent;\n"
            "CREATE TABLE parent (id INTEGER PRIMARY KEY);\n"
        )

        with self.assertRaises(sqlite3.IntegrityError):
            apply_migrations(self.db_path, migrations_dir=migrations_dir)

        # 000_setup committed (its own transaction); 001_rebuild rolled back
        # whole, including the schema_migrations insert it never reached.
        with db.connect(self.db_path) as conn:
            recorded = {r["filename"] for r in conn.execute("SELECT filename FROM schema_migrations")}
        self.assertIn("000_setup.sql", recorded)
        self.assertNotIn("001_rebuild.sql", recorded)

    def test_no_transaction_marker_disables_foreign_keys(self) -> None:
        """Same rebuild, marked no-transaction: it must succeed."""
        migrations_dir = self._write_fixture(
            "-- migrate: no-transaction\n"
            "PRAGMA foreign_keys = OFF;\n"
            "DROP TABLE parent;\n"
            "CREATE TABLE parent (id INTEGER PRIMARY KEY);\n"
            "PRAGMA foreign_keys = ON;\n"
        )

        applied = apply_migrations(self.db_path, migrations_dir=migrations_dir)

        self.assertEqual(applied, ["000_setup.sql", "001_rebuild.sql"])
        with db.connect(self.db_path) as conn:
            recorded = {r["filename"] for r in conn.execute("SELECT filename FROM schema_migrations")}
        self.assertIn("001_rebuild.sql", recorded)


class Migration002TestCase(StorageTestCase):
    def setUp(self) -> None:
        super().setUp()
        apply_migrations(self.db_path)


class TestMigration002Schema(Migration002TestCase):
    def test_forward_migration_creates_all_three_objects(self) -> None:
        tables = table_names(self.db_path)
        self.assertIn("facts", tables)
        self.assertIn("facts_vec", tables)
        self.assertIn("approvals", tables)

    def test_tasks_idempotency_key_is_gone(self) -> None:
        """001c already removed it; 002 does NOT re-issue the drop.

        Re-issuing `ALTER TABLE tasks DROP COLUMN idempotency_key` would fail
        with "no such column" and block the whole migration. The outcome the
        Sprint 2 plan wanted is still asserted here — the column is absent, and
        the key now lives on approvals.
        """
        self.assertNotIn("idempotency_key", task_columns(self.db_path))

    def test_idempotency_key_moved_to_approvals(self) -> None:
        with db.connect(self.db_path) as conn:
            columns = [r["name"] for r in conn.execute("PRAGMA table_info(approvals)")]
        self.assertIn("idempotency_key", columns)

    def test_approvals_keeps_approved_and_executed_separate(self) -> None:
        """The gap between them is the crash window the key exists for.

        A schema that collapsed them would still pass every other test here,
        so the enum is asserted explicitly.
        """
        for status in ("pending", "approved", "rejected", "executed", "failed"):
            with self.subTest(status=status):
                with db.transaction(self.db_path) as conn:
                    insert_run(conn)
                    insert_approval(conn, 1, "key-1", status=status)
                with db.transaction(self.db_path) as conn:
                    conn.execute("DELETE FROM approvals")
                    conn.execute("DELETE FROM runs")


class TestLlmCallsEmbedderRole(Migration002TestCase):
    """002 widens the role CHECK so the memory server can log its own calls.

    role was an ENFORCED CHECK in 001c, not a convention, so this needed a
    table rebuild. These tests pin both halves: the new value is admitted, and
    the constraint still rejects everything outside the enum.
    """

    def setUp(self) -> None:
        super().setUp()
        with db.transaction(self.db_path) as conn:
            insert_run(conn)

    def _insert_call(self, conn, call_id: str, role: str, **overrides) -> None:
        row = {
            "call_id": call_id,
            "run_id": "run-1",
            "role": role,
            "model_tag": "nomic-embed-text",
            "think_mode": "n/a",
        }
        row.update(overrides)
        columns = ", ".join(row)
        placeholders = ", ".join("?" for _ in row)
        conn.execute(
            f"INSERT INTO llm_calls ({columns}) VALUES ({placeholders})",
            tuple(row.values()),
        )

    def test_embedder_role_is_accepted(self) -> None:
        with db.transaction(self.db_path) as conn:
            self._insert_call(conn, "call-embed", "embedder")

        self.assertEqual(count(self.db_path, "llm_calls"), 1)

    def test_bogus_role_is_still_rejected(self) -> None:
        """Unchanged behaviour: the CHECK is widened, not removed.

        Without this, "we added 'embedder'" and "we dropped the constraint"
        would be indistinguishable.
        """
        with self.assertRaises(sqlite3.IntegrityError):
            with db.transaction(self.db_path) as conn:
                self._insert_call(conn, "call-bogus", "not_a_real_role")

    def test_original_five_roles_still_accepted(self) -> None:
        for role in ("planner", "critic", "writer", "judge", "router"):
            with self.subTest(role=role):
                with db.transaction(self.db_path) as conn:
                    self._insert_call(conn, f"call-{role}", role, think_mode="False")

    def test_embedding_row_leaves_output_metrics_null_not_zero(self) -> None:
        """NULL means not applicable; 0 would be a measurement never taken.

        Ollama's embed response genuinely omits eval_count and eval_duration
        (verified against a real nomic-embed-text call), so writing 0 there
        would assert a number nobody measured and would drag any future
        AVG(tok_s) toward zero.
        """
        with db.transaction(self.db_path) as conn:
            self._insert_call(
                conn, "call-embed", "embedder",
                prompt_eval_count=9,
                total_duration_ns=4302553600,
                ok=1,
            )

        with db.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM llm_calls WHERE call_id = ?", ("call-embed",)
            ).fetchone()

        for column in ("eval_count", "eval_duration_ns", "tok_s", "raw_response",
                       "task_id", "step_index", "prompt_file"):
            with self.subTest(column=column):
                self.assertIsNone(row[column])

        self.assertEqual(row["prompt_eval_count"], 9)
        self.assertEqual(row["total_duration_ns"], 4302553600)

    def test_rebuild_preserved_the_locked_18_columns(self) -> None:
        """The rebuild must not have taken the opportunity to change anything."""
        self.assertEqual(llm_calls_columns(self.db_path), EXPECTED_LLM_CALLS_COLUMNS)

    def test_rebuild_recreated_both_indexes(self) -> None:
        with db.connect(self.db_path) as conn:
            names = {
                r["name"]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'index' AND tbl_name = 'llm_calls' "
                    "AND name NOT LIKE 'sqlite_%'"
                )
            }
        self.assertEqual(names, {"idx_llm_calls_run", "idx_llm_calls_task_step"})

    def test_rebuild_preserved_existing_rows(self) -> None:
        """Rows written under the 001c schema must survive the 002 rebuild.

        Built by migrating a 001c-era database forward rather than by inserting
        after the fact, so this exercises the real INSERT ... SELECT copy.
        """
        legacy = Path(self._tmp.name) / "legacy.db"
        apply_only(legacy, "001_initial.sql")
        apply_only(legacy, "001b_task_columns.sql")
        apply_only(legacy, "001c_lock_schema.sql")

        with db.transaction(legacy) as conn:
            insert_run(conn)
            conn.execute(
                "INSERT INTO llm_calls (call_id, run_id, step_index, role, model_tag, "
                "think_mode, prompt_eval_count, eval_count, eval_duration_ns, tok_s, ok) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("call-legacy", "run-1", 3, "planner", "qwen3.5:9b", "False",
                 110, 88, 4_000_000_000, 22.0, 1),
            )

        self.assertEqual(apply_migrations(legacy), ["002_facts_approvals.sql"])

        with db.connect(legacy) as conn:
            row = conn.execute(
                "SELECT * FROM llm_calls WHERE call_id = ?", ("call-legacy",)
            ).fetchone()
            self.assertEqual(conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            self.assertEqual(conn.execute("PRAGMA foreign_key_check").fetchall(), [])

        self.assertEqual(row["role"], "planner")
        self.assertEqual(row["eval_count"], 88)
        self.assertEqual(row["tok_s"], 22.0)
        self.assertEqual(row["step_index"], 3)


class TestApprovalsConstraints(Migration002TestCase):
    def setUp(self) -> None:
        super().setUp()
        with db.transaction(self.db_path) as conn:
            insert_run(conn)

    def test_duplicate_idempotency_key_is_rejected(self) -> None:
        """The actual protection against double-firing an external action."""
        with db.transaction(self.db_path) as conn:
            insert_approval(conn, 1, "run-1:task-0:send_email")

        with self.assertRaises(sqlite3.IntegrityError):
            with db.transaction(self.db_path) as conn:
                insert_approval(conn, 2, "run-1:task-0:send_email")

        self.assertEqual(count(self.db_path, "approvals"), 1)

    def test_invalid_status_is_rejected(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            with db.transaction(self.db_path) as conn:
                insert_approval(conn, 1, "key-1", status="not_a_real_status")


class TestFactsInvariants(Migration002TestCase):
    def setUp(self) -> None:
        super().setUp()
        with db.transaction(self.db_path) as conn:
            insert_run(conn)

    def test_second_current_row_for_same_entity_is_rejected(self) -> None:
        """THE invariant: one current row per (target_id, entity_key).

        Enforced by the partial unique index, not by application code
        remembering to flip the old row first.
        """
        with db.transaction(self.db_path) as conn:
            insert_fact(conn, 1, "intake_dates", is_current=1)

        with self.assertRaises(sqlite3.IntegrityError):
            with db.transaction(self.db_path) as conn:
                insert_fact(conn, 2, "intake_dates", is_current=1)

        self.assertEqual(count(self.db_path, "facts"), 1)

    def test_superseded_rows_are_exempt_from_the_unique_index(self) -> None:
        """The other half: history must be able to hold many versions.

        A plain (non-partial) unique index would pass the test above and fail
        this one, so the pair pins down which index was actually created.
        """
        with db.transaction(self.db_path) as conn:
            insert_fact(conn, 1, "intake_dates", is_current=0, version=1)
            insert_fact(conn, 2, "intake_dates", is_current=0, version=2)
            insert_fact(
                conn, 3, "intake_dates", is_current=1, version=3,
                supersedes_id=2, change_kind="modified",
            )

        self.assertEqual(count(self.db_path, "facts"), 3)

    def test_invalid_change_kind_is_rejected(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            with db.transaction(self.db_path) as conn:
                insert_fact(conn, 1, "intake_dates", change_kind="invented")


class TestFactsVec(Migration002TestCase):
    """vec0 mechanics only. Vectors are random — the embedding module is next."""

    def test_accepts_a_768_float_vector(self) -> None:
        with db.transaction(self.db_path) as conn:
            conn.execute(
                "INSERT INTO facts_vec (fact_id, embedding) VALUES (?, ?)",
                (1, random_vector()),
            )
        self.assertEqual(count(self.db_path, "facts_vec"), 1)

    def test_rejects_a_767_float_vector(self) -> None:
        """Width is fixed by nomic-embed-text. A wrong-width vector must not
        land silently — OperationalError, not IntegrityError: vec0 reports a
        dimension mismatch from the virtual table, not a constraint.
        """
        with self.assertRaises(sqlite3.OperationalError):
            with db.transaction(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO facts_vec (fact_id, embedding) VALUES (?, ?)",
                    (1, random_vector(767)),
                )

    def test_knn_over_three_vectors_returns_three_in_distance_order(self) -> None:
        with db.transaction(self.db_path) as conn:
            for fact_id in (1, 2, 3):
                conn.execute(
                    "INSERT INTO facts_vec (fact_id, embedding) VALUES (?, ?)",
                    (fact_id, random_vector()),
                )

        with db.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT fact_id, distance FROM facts_vec "
                "WHERE embedding MATCH ? ORDER BY distance LIMIT 3",
                (random_vector(),),
            ).fetchall()

        self.assertEqual(len(rows), 3)
        distances = [r["distance"] for r in rows]
        self.assertEqual(distances, sorted(distances))
        self.assertEqual({r["fact_id"] for r in rows}, {1, 2, 3})

    def test_knn_over_an_empty_table_returns_nothing_and_does_not_raise(self) -> None:
        """NEGATIVE CONTROL for the KNN test above.

        A positive-only suite proves nothing: a query that raised on empty, or
        one that returned rows from somewhere else, would be invisible. The
        memory server queries before it has written anything on the very first
        run of a new target, so this is the real first-run path, not a
        hypothetical.
        """
        self.assertEqual(count(self.db_path, "facts_vec"), 0)

        with db.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT fact_id, distance FROM facts_vec "
                "WHERE embedding MATCH ? ORDER BY distance LIMIT 3",
                (random_vector(),),
            ).fetchall()

        self.assertEqual(rows, [])


class TestConnectionHelper(Migration002TestCase):
    """There must be exactly one way to open this database."""

    def test_connect_applies_every_required_pragma(self) -> None:
        with db.connect(self.db_path) as conn:
            self.assertEqual(
                conn.execute("PRAGMA journal_mode").fetchone()[0].lower(), "wal"
            )
            self.assertEqual(conn.execute("PRAGMA busy_timeout").fetchone()[0], 5000)
            self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)

    def test_connect_loads_sqlite_vec(self) -> None:
        """Without this, migration 002 dies on "no such module: vec0"."""
        with db.connect(self.db_path) as conn:
            version = conn.execute("SELECT vec_version()").fetchone()[0]
        self.assertTrue(version.startswith("v0.1"), version)


class TestBeginImmediate(Migration002TestCase):
    """Matched pair proving `atomic()` uses BEGIN IMMEDIATE, and why it matters.

    Both halves run two connections in ONE thread with a deliberately held
    lock, so neither depends on scheduling luck.
    """

    #: Short enough to keep the test fast, long enough that "waited" and
    #: "failed instantly" cannot be confused for each other.
    BUSY_MS = 1000

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None)
        conn.execute(f"PRAGMA busy_timeout = {self.BUSY_MS}")
        return conn

    @contextmanager
    def _write_lock_held(self):
        """Hold the database write lock, uncommitted, for the block's duration."""
        holder = self._connection()
        holder.execute("BEGIN IMMEDIATE")
        holder.execute(
            "INSERT INTO runs (run_id, target_id, status) VALUES (?, ?, ?)",
            ("run-holder", "t", "running"),
        )
        try:
            yield
        finally:
            holder.execute("ROLLBACK")
            holder.close()

    def test_deferred_begin_ignores_busy_timeout_negative_control(self) -> None:
        """The bug, demonstrated: a plain BEGIN fails INSTANTLY under contention.

        A deferred transaction takes no lock until its first statement, so a
        read-then-write has to upgrade a read lock to a write lock partway
        through. busy_timeout does not apply to that upgrade — SQLite returns
        SQLITE_BUSY immediately, because waiting could deadlock two readers
        both trying to upgrade. Measured here at ~0.000s against a 1000ms
        timeout.

        This is the negative control for the test below: without it, "we use
        BEGIN IMMEDIATE" is an unverified claim in a docstring.
        """
        with self._write_lock_held():
            other = self._connection()
            other.execute("BEGIN")  # deferred — the bug
            other.execute("SELECT COUNT(*) FROM runs").fetchone()  # read lock

            start = time.perf_counter()
            with self.assertRaises(sqlite3.OperationalError):
                other.execute(
                    "INSERT INTO runs (run_id, target_id, status) VALUES (?, ?, ?)",
                    ("run-other", "t", "running"),
                )
            elapsed = time.perf_counter() - start

            try:
                other.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            other.close()

        self.assertLess(
            elapsed,
            self.BUSY_MS / 1000 / 2,
            f"deferred upgrade waited {elapsed:.3f}s; it should fail instantly",
        )

    def test_begin_immediate_waits_for_the_busy_timeout(self) -> None:
        """The fix: BEGIN IMMEDIATE takes the write lock up front, so
        contention becomes a bounded WAIT instead of an instant error.

        It still raises here, because the lock is held for the whole timeout on
        purpose. The assertion is on the waiting, which is the actual
        difference between the two code paths.
        """
        with self._write_lock_held():
            other = self._connection()
            start = time.perf_counter()
            with self.assertRaises(sqlite3.OperationalError):
                other.execute("BEGIN IMMEDIATE")
            elapsed = time.perf_counter() - start
            other.close()

        self.assertGreater(
            elapsed,
            self.BUSY_MS / 1000 / 2,
            f"BEGIN IMMEDIATE failed after only {elapsed:.3f}s; it should have waited",
        )

    def test_atomic_uses_begin_immediate(self) -> None:
        """Pins the helper itself, not just the raw SQL above.

        If `atomic()` regressed to a plain BEGIN, the write below would fail
        instantly rather than after the timeout.
        """
        with self._write_lock_held():
            conn = self._connection()
            start = time.perf_counter()
            with self.assertRaises(sqlite3.OperationalError):
                with db.atomic(conn):
                    conn.execute(
                        "INSERT INTO runs (run_id, target_id, status) VALUES (?, ?, ?)",
                        ("run-atomic", "t", "running"),
                    )
            elapsed = time.perf_counter() - start
            conn.close()

        self.assertGreater(elapsed, self.BUSY_MS / 1000 / 2, f"{elapsed:.3f}s")


class TestConcurrentWriters(Migration002TestCase):
    WRITERS = 2
    WRITES_EACH = 100

    def test_two_writers_land_every_row_with_no_busy_errors(self) -> None:
        """The two-process model from CLAUDE.md, in miniature.

        Each writer opens its OWN connection through db.connect() — so the real
        pragmas and the real BEGIN IMMEDIATE path are what is under test, not a
        hand-rolled connection that happens to be configured correctly.
        """
        errors: "list[str]" = []
        errors_lock = threading.Lock()

        def writer(worker_id: int) -> None:
            try:
                with db.connect(self.db_path) as conn:
                    for i in range(self.WRITES_EACH):
                        try:
                            with db.atomic(conn):
                                conn.execute(
                                    "INSERT INTO runs (run_id, target_id, status) "
                                    "VALUES (?, ?, ?)",
                                    (f"run-{worker_id}-{i}", "t", "running"),
                                )
                        except sqlite3.OperationalError as exc:
                            with errors_lock:
                                errors.append(f"worker {worker_id} iter {i}: {exc!r}")
            except Exception as exc:  # noqa: BLE001 - must surface, never swallow
                with errors_lock:
                    errors.append(f"worker {worker_id} died: {exc!r}")

        threads = [threading.Thread(target=writer, args=(w,)) for w in range(self.WRITERS)]
        start = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        elapsed = time.perf_counter() - start

        self.assertFalse([t for t in threads if t.is_alive()], "a writer thread hung")
        self.assertEqual(errors, [], f"OperationalError(s) after {elapsed:.2f}s")
        self.assertEqual(count(self.db_path, "runs"), self.WRITERS * self.WRITES_EACH)


if __name__ == "__main__":
    unittest.main(verbosity=2)
