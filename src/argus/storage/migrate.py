"""Apply pending .sql migrations, in filename order, each in its own transaction.

Run: python -m argus.storage.migrate    (with src/ on PYTHONPATH)
     python src/argus/storage/migrate.py

No-transaction marker:
  A migration that needs `PRAGMA foreign_keys = OFF` to take effect (e.g. a
  table rebuild while another table still holds foreign-key-referencing rows)
  must run outside a transaction — the pragma is a silent no-op mid-transaction,
  confirmed empirically: it reads back unchanged and the FK check still fires.
  Put `-- migrate: no-transaction` as the literal FIRST LINE of the .sql file to
  opt out of the per-file transaction. The file must then re-enable
  `PRAGMA foreign_keys = ON` itself before it ends. Trade-off: no-transaction
  migrations lose the all-or-nothing guarantee — a failure partway leaves the
  schema changed but NOT recorded in schema_migrations, so a re-run will retry
  from the top and likely hit "already exists" errors. Use only when a rebuild
  genuinely needs FK enforcement off; every other migration should stay
  transactional.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List

if __package__ in (None, ""):  # direct-script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from argus.storage import db  # noqa: E402

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
NO_TRANSACTION_MARKER = "-- migrate: no-transaction"

# schema_migrations is also created by 001_initial.sql; this is the bootstrap
# copy so the "which migrations have run?" query works on an empty database.
_BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename    TEXT PRIMARY KEY,
    applied_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
)
"""


def _statements(sql: str) -> Iterator[str]:
    """Split a script into complete SQL statements.

    Uses sqlite3.complete_statement rather than splitting on ';' so semicolons
    inside string literals and comments do not split a statement. executescript()
    is deliberately not used: it COMMITs any open transaction first, which would
    break the one-transaction-per-migration guarantee.
    """
    buffer = ""
    for line in sql.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            statement = buffer.strip()
            if statement:
                yield statement
            buffer = ""
    tail = buffer.strip()
    if tail:
        yield tail


def _is_no_transaction(sql: str) -> bool:
    """True if the migration opts out of its per-file transaction.

    Requires the marker as the literal first line — no leading blank lines or
    other comments — so the check is unambiguous rather than a heuristic scan.
    """
    first_line = sql.splitlines()[0].strip() if sql.strip() else ""
    return first_line == NO_TRANSACTION_MARKER


#: How long a second migrator waits for the first before giving up.
LOCK_TIMEOUT_S = 120.0


@contextmanager
def _migration_lock(db_path: db.PathLike) -> Iterator[None]:
    """Serialise migrators across processes with an OS file lock.

    Review H-1: re-checking schema_migrations inside each file's transaction
    was not enough. Two processes opening a FRESH file race on
    `PRAGMA journal_mode = WAL` inside db.connect() and the loser fails with
    "database is locked" before any re-check runs; and the no-transaction
    path (001c) cannot be protected by a transaction at all. One lock around
    the whole migration, connect included, covers both.
    """
    lock_path = Path(str(db_path) + ".migrate.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+b")
    deadline = time.monotonic() + LOCK_TIMEOUT_S
    try:
        while True:
            try:
                if sys.platform == "win32":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"another process holds {lock_path}")
                time.sleep(0.05)
        try:
            yield
        finally:
            if sys.platform == "win32":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def apply_migrations(
    db_path: db.PathLike = db.DB_PATH, migrations_dir: Path = MIGRATIONS_DIR
) -> List[str]:
    """Apply every migration not yet recorded. Returns the filenames applied."""
    with _migration_lock(db_path):
        return _apply_locked(db_path, migrations_dir)


def _apply_locked(db_path: db.PathLike, migrations_dir: Path) -> List[str]:
    applied: List[str] = []

    with db.connect(db_path) as conn:
        conn.execute(_BOOTSTRAP)
        already = {row["filename"] for row in conn.execute("SELECT filename FROM schema_migrations")}

        for path in sorted(migrations_dir.glob("*.sql")):
            if path.name in already:
                continue

            sql = path.read_text(encoding="utf-8")

            if _is_no_transaction(sql):
                for statement in _statements(sql):
                    conn.execute(statement)
                conn.execute(
                    "INSERT INTO schema_migrations (filename) VALUES (?)", (path.name,)
                )
            else:
                with db.atomic(conn):
                    # Audit M5: re-check under the write lock. `already` was
                    # read before any lock, so a second process starting at the
                    # same moment would otherwise re-run a file the first one
                    # just applied and die on "table already exists".
                    if conn.execute(
                        "SELECT 1 FROM schema_migrations WHERE filename = ?", (path.name,)
                    ).fetchone():
                        continue
                    for statement in _statements(sql):
                        conn.execute(statement)
                    conn.execute(
                        "INSERT INTO schema_migrations (filename) VALUES (?)", (path.name,)
                    )
            applied.append(path.name)

    return applied


def main() -> None:
    applied = apply_migrations()
    if applied:
        for filename in applied:
            print(f"applied {filename}")
    else:
        print("no pending migrations")
    print(f"database: {db.DB_PATH}")


if __name__ == "__main__":
    main()
