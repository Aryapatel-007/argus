"""Storage layer tests.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from argus.storage import db  # noqa: E402
from argus.storage.migrate import MIGRATIONS_DIR, apply_migrations  # noqa: E402

EXPECTED_TABLES = {"schema_migrations", "runs", "tasks", "checkpoints", "llm_calls"}
EXPECTED_MIGRATIONS = ["001_initial.sql", "001b_task_columns.sql"]


def task_columns(db_path: Path) -> list:
    with db.connect(db_path) as conn:
        return [row["name"] for row in conn.execute("PRAGMA table_info(tasks)")]


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


class TestIdempotencyKeyColumn(StorageTestCase):
    def test_fresh_db_has_idempotency_key(self) -> None:
        apply_migrations(self.db_path)
        self.assertIn("idempotency_key", task_columns(self.db_path))

    def test_upgrade_from_001_only_adds_the_column(self) -> None:
        """The path that can break silently: a database that already ran 001.

        Amending 001 in place would leave this database untouched, because
        migrate.py skips by filename. 001b is what makes the upgrade happen.
        """
        apply_only(self.db_path, "001_initial.sql")
        self.assertNotIn("idempotency_key", task_columns(self.db_path))

        with db.transaction(self.db_path) as conn:
            insert_run_task_checkpoint(conn)

        self.assertEqual(apply_migrations(self.db_path), ["001b_task_columns.sql"])
        self.assertIn("idempotency_key", task_columns(self.db_path))

        # Pre-existing rows survive the upgrade and default to NULL.
        with db.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT task_id, idempotency_key FROM tasks WHERE task_id = ?",
                ("task-1",),
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertIsNone(row["idempotency_key"])

    def test_key_round_trips(self) -> None:
        apply_migrations(self.db_path)
        with db.transaction(self.db_path) as conn:
            insert_run_task_checkpoint(conn)
            conn.execute(
                "UPDATE tasks SET idempotency_key = ? WHERE task_id = ?",
                ("run-1:task-1:act:0", "task-1"),
            )

        with db.connect(self.db_path) as conn:
            key = conn.execute(
                "SELECT idempotency_key FROM tasks WHERE task_id = ?", ("task-1",)
            ).fetchone()["idempotency_key"]
        self.assertEqual(key, "run-1:task-1:act:0")


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
