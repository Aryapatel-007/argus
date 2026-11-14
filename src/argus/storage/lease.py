"""Run leases: exactly one process drives a run at a time (audit H2).

A lease row names its owner as "<host>:<pid>:<random>". The owner proves it
still holds the lease by updating `heartbeat_at` INSIDE every step
transaction, with a rowcount check (`assert_held`). A process that lost the
lease therefore cannot commit even one step: its transaction raises and rolls
back.

Takeover rules (`acquire`):
  - no lease row            -> take it
  - same owner              -> refresh it
  - owner is on this host and its pid is dead  -> take it at once
    (the common case: resume right after a crash)
  - owner is on this host and its pid is alive -> refuse, even if the
    heartbeat is stale (a single step can outlast any fixed window)
  - owner on another host, heartbeat older than `stale_after_s` -> take it
  - otherwise               -> refuse

Liveness is checked with OpenProcess/GetExitCodeProcess on Windows. Never
`os.kill(pid, 0)` there: on Windows os.kill calls TerminateProcess, so a
"liveness check" with signal 0 would kill the process it was asking about.
"""

from __future__ import annotations

import os
import socket
import sqlite3
import sys
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from argus.storage import db

DEFAULT_STALE_AFTER_S = 120.0


class LeaseLost(RuntimeError):
    """This process no longer owns the run. Stop driving it."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def new_owner() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            # Only "no such process" means dead. ERROR_ACCESS_DENIED (an owner
            # running elevated or as another account) means it EXISTS; calling
            # that dead would hand its live run to someone else (review M-1).
            ERROR_INVALID_PARAMETER = 87
            return ctypes.get_last_error() != ERROR_INVALID_PARAMETER
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)  # security-scan: ok POSIX branch only; signal 0 is a pure liveness probe there
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _owner_liveness(owner: str) -> Optional[bool]:
    """True/False when the owner is on this host and we can tell; None otherwise."""
    try:
        host, pid, _ = owner.split(":", 2)
        if host != socket.gethostname():
            return None
        return pid_alive(int(pid))
    except (ValueError, TypeError):
        return None


def acquire(
    conn: sqlite3.Connection,
    run_id: str,
    owner: str,
    *,
    stale_after_s: float = DEFAULT_STALE_AFTER_S,
) -> bool:
    """Take or refresh the lease. True if `owner` now holds it."""
    now = _now()
    with db.atomic(conn):
        row = conn.execute(
            "SELECT owner, heartbeat_at FROM run_leases WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO run_leases (run_id, owner, acquired_at, heartbeat_at) "
                "VALUES (?, ?, ?, ?)",
                (run_id, owner, _fmt(now), _fmt(now)),
            )
            return True
        stale = now - _parse(row["heartbeat_at"]) > timedelta(seconds=stale_after_s)
        alive = _owner_liveness(row["owner"])
        # Same host: the pid is the truth. A live owner keeps its lease however
        # stale its heartbeat looks, because one step can legitimately outlast
        # the stale window (a planner schema retry is 2 x 120s; review H-2).
        # Staleness only decides when liveness is unknowable (another host).
        if row["owner"] == owner or alive is False or (alive is None and stale):
            conn.execute(
                "UPDATE run_leases SET owner = ?, acquired_at = ?, heartbeat_at = ? "
                "WHERE run_id = ?",
                (owner, _fmt(now), _fmt(now), run_id),
            )
            return True
        return False


def assert_held(conn: sqlite3.Connection, run_id: str, owner: str) -> None:
    """Heartbeat + ownership check. Call INSIDE the step's transaction."""
    cursor = conn.execute(
        "UPDATE run_leases SET heartbeat_at = ? WHERE run_id = ? AND owner = ?",
        (_fmt(_now()), run_id, owner),
    )
    if cursor.rowcount != 1:
        raise LeaseLost(f"run {run_id} is not leased to {owner}")


def insert_initial(conn: sqlite3.Connection, run_id: str, owner: str) -> None:
    """For a brand-new run: called in the same transaction that inserts runs."""
    now = _fmt(_now())
    conn.execute(
        "INSERT INTO run_leases (run_id, owner, acquired_at, heartbeat_at) VALUES (?, ?, ?, ?)",
        (run_id, owner, now, now),
    )


def release(conn: sqlite3.Connection, run_id: str, owner: str) -> None:
    with db.atomic(conn):
        conn.execute("DELETE FROM run_leases WHERE run_id = ? AND owner = ?", (run_id, owner))


def current_owner(conn: sqlite3.Connection, run_id: str) -> Optional[str]:
    row = conn.execute("SELECT owner FROM run_leases WHERE run_id = ?", (run_id,)).fetchone()
    return row["owner"] if row else None
