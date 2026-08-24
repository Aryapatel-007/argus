"""SQLite connection and transaction helpers for Argus.

WAL mode, foreign keys on, busy timeout set, sqlite-vec loaded, explicit
transaction control.

This is the ONLY way to open the Argus database. Both processes that touch the
file — the orchestrator and the memory server — go through `connect()`, so the
pragmas and the extension are never a thing someone remembered to do.

The orchestrator's crash-recovery guarantee depends on `atomic()`/`transaction()`:
a state transition and its checkpoint insert go inside ONE of these blocks, so a
crash mid-step leaves the database exactly as it was before the step started.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Union

import sqlite_vec

# src/argus/storage/db.py -> parents[3] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_ROOT / "data" / "argus.db"

PathLike = Union[str, Path]

#: Driver-level wait for the file lock, in seconds. Belt to busy_timeout's
#: braces: this one covers the connect itself, busy_timeout covers statements.
CONNECT_TIMEOUT_SECONDS = 5.0

_PRAGMAS = (
    "PRAGMA journal_mode = WAL",
    # Two writer processes share this file. Without this, a writer that finds
    # the lock held gets SQLITE_BUSY immediately instead of waiting.
    "PRAGMA busy_timeout = 5000",
    "PRAGMA foreign_keys = ON",
    "PRAGMA synchronous = NORMAL",
)


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Load the vec0 extension. Required before migration 002 can run.

    `sqlite_vec.load(conn)`, never a hand-rolled `conn.load_extension()`:
    `sqlite_vec.loadable_path()` returns an extensionless path
    (`...\\sqlite_vec\\vec0`) which does not exist on disk — only `vec0.dll`
    does. SQLite's own loader appends the platform suffix, which is why the
    bare path works when passed through `load()` and why passing
    `loadable_path()` around by hand is a trap.

    Extension loading is re-disabled immediately afterwards: it is only needed
    for this one call, and leaving it on lets any later SQL in the process load
    arbitrary native code.
    """
    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)


@contextmanager
def connect(db_path: PathLike = DB_PATH) -> Iterator[sqlite3.Connection]:
    """Open a connection with the Argus pragmas and sqlite-vec applied.

    Closes on exit.

    isolation_level=None puts the driver in autocommit mode, so transactions are
    started and ended explicitly here rather than implicitly by the sqlite3
    module. Without that, transaction boundaries depend on statement type and
    are not something the checkpoint guarantee can be built on — and, more
    sharply, the driver would issue a plain deferred BEGIN, which is the bug
    `atomic()` documents below.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        path, timeout=CONNECT_TIMEOUT_SECONDS, isolation_level=None
    )
    conn.row_factory = sqlite3.Row
    try:
        for pragma in _PRAGMAS:
            conn.execute(pragma)
        _load_sqlite_vec(conn)
        yield conn
    finally:
        conn.close()


@contextmanager
def atomic(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """One atomic unit of work on an existing connection. EVERY write uses this.

    Commits on clean exit, rolls back on any exception (including
    KeyboardInterrupt and SystemExit) and re-raises.

    BEGIN IMMEDIATE, not BEGIN — and the difference is a real bug, not a style
    preference. A plain `BEGIN` starts a DEFERRED transaction, which takes no
    lock at all until its first statement. A transaction that reads and then
    writes therefore has to UPGRADE from a read lock to a write lock partway
    through. `busy_timeout` does not apply to that upgrade: if another
    connection holds the write lock, SQLite returns SQLITE_BUSY instantly,
    however long the timeout is set to, because waiting could deadlock two
    readers each trying to upgrade. The failure is invisible under a single
    writer and shows up only once the memory server and the orchestrator write
    concurrently — as intermittent OperationalError under load, which is the
    worst possible time to discover it.

    BEGIN IMMEDIATE takes the write lock up front, before any statement runs.
    Contention becomes a wait bounded by busy_timeout instead of an instant
    error.

    Corollary, enforced by convention rather than by code: never hold one of
    these open across an Ollama call. An embedding call can take a second; the
    write lock is database-wide. Embed first, then open the transaction.
    """
    conn.execute("BEGIN IMMEDIATE")
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
