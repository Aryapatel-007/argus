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


def apply_migrations(
    db_path: db.PathLike = db.DB_PATH, migrations_dir: Path = MIGRATIONS_DIR
) -> List[str]:
    """Apply every migration not yet recorded. Returns the filenames applied."""
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
