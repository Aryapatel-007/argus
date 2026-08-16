"""Apply pending .sql migrations, in filename order, each in its own transaction.

Run: python -m argus.storage.migrate    (with src/ on PYTHONPATH)
     python src/argus/storage/migrate.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Iterator, List

if __package__ in (None, ""):  # direct-script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from argus.storage import db  # noqa: E402

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

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


def apply_migrations(db_path: db.PathLike = db.DB_PATH) -> List[str]:
    """Apply every migration not yet recorded. Returns the filenames applied."""
    applied: List[str] = []

    with db.connect(db_path) as conn:
        conn.execute(_BOOTSTRAP)
        already = {row["filename"] for row in conn.execute("SELECT filename FROM schema_migrations")}

        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in already:
                continue

            sql = path.read_text(encoding="utf-8")
            with db.atomic(conn):
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
