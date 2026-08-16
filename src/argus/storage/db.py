"""SQLite connection and transaction helpers for Argus.

WAL mode, foreign keys on, explicit transaction control.

The orchestrator's crash-recovery guarantee depends on `atomic()`/`transaction()`:
a state transition and its checkpoint insert go inside ONE of these blocks, so a
crash mid-step leaves the database exactly as it was before the step started.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Union

# src/argus/storage/db.py -> parents[3] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_ROOT / "data" / "argus.db"

PathLike = Union[str, Path]

_PRAGMAS = (
    "PRAGMA journal_mode = WAL",
    "PRAGMA foreign_keys = ON",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA busy_timeout = 5000",
)


@contextmanager
def connect(db_path: PathLike = DB_PATH) -> Iterator[sqlite3.Connection]:
    """Open a connection with the Argus pragmas applied. Closes on exit.

    isolation_level=None puts the driver in autocommit mode, so transactions are
    started and ended explicitly here rather than implicitly by the sqlite3
    module. Without that, transaction boundaries depend on statement type and
    are not something the checkpoint guarantee can be built on.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        for pragma in _PRAGMAS:
            conn.execute(pragma)
        yield conn
    finally:
        conn.close()


@contextmanager
def atomic(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """One atomic unit of work on an existing connection.

    Commits on clean exit, rolls back on any exception (including
    KeyboardInterrupt and SystemExit) and re-raises.
    """
    conn.execute("BEGIN")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


@contextmanager
def transaction(db_path: PathLike = DB_PATH) -> Iterator[sqlite3.Connection]:
    """Open a connection and run one atomic unit of work on it."""
    with connect(db_path) as conn:
        with atomic(conn) as tx:
            yield tx
