"""Storage layer tests.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from argus.storage import db  # noqa: E402
from argus.storage.migrate import MIGRATIONS_DIR, apply_migrations  # noqa: E402

EXPECTED_TABLES = {"schema_migrations", "runs", "tasks", "checkpoints", "llm_calls"}
EXPECTED_MIGRATIONS = ["001_initial.sql", "001b_task_columns.sql", "001c_lock_schema.sql"]

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
    def test_fresh_db_gets_all_five_tables(self) -> None:
        applied = apply_migrations(self.db_path)

        self.assertEqual(applied, EXPECTED_MIGRATIONS)
        self.assertTrue(EXPECTED_TABLES.issubset(table_names(self.db_path)))

    def test_migrations_apply_in_filename_order(self) -> None:
        """001b must sort after 001 and before any future 002."""
        found = sorted(p.name for p in MIGRATIONS_DIR.glob("*.sql"))
        self.assertEqual(found, EXPECTED_MIGRATIONS)

    def test_wal_mode_is_on(self) -> None:
        apply_migrations(self.db_path)
        with db.connect(self.db_path) as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(mode.lower(), "wal")

    def test_rerunning_applies_nothing(self) -> None:
        apply_migrations(self.db_path)
        self.assertEqual(apply_migrations(self.db_path), [])
        self.assertEqual(count(self.db_path, "schema_migrations"), len(EXPECTED_MIGRATIONS))


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

        self.assertEqual(apply_migrations(self.db_path), ["001c_lock_schema.sql"])
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
