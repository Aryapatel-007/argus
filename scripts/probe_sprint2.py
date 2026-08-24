"""Sprint 2 environment probe.

Checks the pieces Sprint 2 will build on top of: sqlite-vec loads and does a
768-dim KNN round trip (facts table needs 768 dims per nomic-embed-text),
WAL survives two concurrent writers, the planner/chat model and the
embedding model can stay resident together, Node is new enough for MCP
Inspector v2, and the repo has the directories Sprint 2 will write into.

Not a server module - print/stdout is deliberate here.

Run:
    python scripts/probe_sprint2.py

This process also doubles as its own WAL-writer subprocess:
    python scripts/probe_sprint2.py --wal-worker <db_path> <worker_id>
"""

from __future__ import annotations

import math
import os
import random
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path


def _wal_worker(db_path: str, worker_id: str, iterations: int = 200) -> None:
    """Runs as a separate OS process against a shared WAL-mode db file."""
    conn = sqlite3.connect(db_path, timeout=5.0, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=5000")
    errors = 0
    for i in range(iterations):
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO wal_test (worker, i) VALUES (?, ?)", (int(worker_id), i)
            )
            conn.execute("COMMIT")
        except sqlite3.OperationalError as exc:
            errors += 1
            print(f"WAL_WORKER_ERROR worker={worker_id} i={i} error={exc!r}", file=sys.stderr)
            try:
                conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
    conn.close()
    print(f"WAL_WORKER_RESULT worker={worker_id} errors={errors}")


if len(sys.argv) > 1 and sys.argv[1] == "--wal-worker":
    _wal_worker(sys.argv[2], sys.argv[3])
    sys.exit(0)


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

THIS_FILE = str(Path(__file__).resolve())
RESULTS: "list[tuple[str, bool]]" = []


def record(name: str, ok: bool) -> None:
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# 1. Python and SQLite
# ---------------------------------------------------------------------------


def check_python_sqlite() -> None:
    section("1. Python and SQLite")
    print(f"sys.version = {sys.version}")
    print(f"sqlite3.sqlite_version = {sqlite3.sqlite_version}")

    ver = tuple(int(p) for p in sqlite3.sqlite_version.split("."))
    record("sqlite_version >= 3.41 (sqlite-vec recommendation)", ver >= (3, 41))
    record("sqlite_version >= 3.35 (ALTER TABLE DROP COLUMN)", ver >= (3, 35))

    has_extension_loading = hasattr(sqlite3.Connection, "enable_load_extension")
    print(f"hasattr(sqlite3.Connection, 'enable_load_extension') = {has_extension_loading}")
    record("sqlite3.Connection has enable_load_extension", has_extension_loading is True)


# ---------------------------------------------------------------------------
# 2. sqlite-vec loads
# ---------------------------------------------------------------------------


def check_sqlite_vec_loads():
    section("2. sqlite-vec loads")
    try:
        import sqlite_vec
    except Exception:
        print(traceback.format_exc())
        record("sqlite-vec import", False)
        return None

    loadable = sqlite_vec.loadable_path()
    print(f"sqlite_vec.loadable_path() = {loadable}")
    print(f"exists (bare path)         = {Path(loadable).exists()}")
    print(f"exists (+ .dll)            = {Path(loadable + '.dll').exists()}")

    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)

    method_used = None

    try:
        sqlite_vec.load(conn)
        method_used = "sqlite_vec.load(conn)"
    except Exception as exc:
        print(f"sqlite_vec.load(conn) failed: {exc!r}")
        print(traceback.format_exc())

        try:
            conn.load_extension(loadable)
            method_used = "(a) conn.load_extension(loadable_path())"
        except Exception as exc_a:
            print(f"fallback (a) failed: {exc_a!r}")
            print(traceback.format_exc())

            try:
                conn.load_extension(loadable + ".dll")
                method_used = "(b) conn.load_extension(loadable_path() + '.dll')"
            except Exception as exc_b:
                print(f"fallback (b) failed: {exc_b!r}")
                print(traceback.format_exc())

                try:
                    os.add_dll_directory(str(Path(loadable).parent))
                    conn.load_extension(loadable)
                    method_used = "(c) os.add_dll_directory() then conn.load_extension()"
                except Exception as exc_c:
                    print(f"fallback (c) failed: {exc_c!r}")
                    print(traceback.format_exc())

    if method_used is None:
        record("sqlite-vec loads", False)
        conn.close()
        return None

    print(f"loaded via: {method_used}")
    try:
        version = conn.execute("select vec_version()").fetchone()[0]
        print(f"vec_version() = {version}")
    except Exception:
        print(traceback.format_exc())
        record("sqlite-vec loads", False)
        conn.close()
        return None

    record("sqlite-vec loads", True)
    return conn


# ---------------------------------------------------------------------------
# 3. 768-dim round trip
# ---------------------------------------------------------------------------


def check_768_round_trip(conn) -> None:
    section("3. 768-dim round trip")
    if conn is None:
        print("skipped: sqlite-vec did not load in check 2")
        record("768-dim KNN round trip", False)
        return

    import sqlite_vec

    try:
        conn.execute(
            "CREATE VIRTUAL TABLE t USING vec0(id INTEGER PRIMARY KEY, embedding FLOAT[768])"
        )
        vectors = [[random.random() for _ in range(768)] for _ in range(3)]
        for i, vec in enumerate(vectors, start=1):
            conn.execute(
                "INSERT INTO t(id, embedding) VALUES (?, ?)",
                (i, sqlite_vec.serialize_float32(vec)),
            )
        query_vec = [random.random() for _ in range(768)]
        rows = conn.execute(
            "SELECT id, distance FROM t WHERE embedding MATCH ? ORDER BY distance LIMIT 3",
            (sqlite_vec.serialize_float32(query_vec),),
        ).fetchall()
    except Exception:
        print(traceback.format_exc())
        record("768-dim KNN round trip", False)
        return

    print(f"returned rows (id, distance) = {[(r[0], r[1]) for r in rows]}")
    distances = [r[1] for r in rows]
    all_finite = all(math.isfinite(d) for d in distances)
    non_identical = len(set(distances)) > 1
    print(
        f"3 rows returned = {len(rows) == 3}, all finite = {all_finite}, "
        f"non-identical = {non_identical}"
    )
    record("768-dim KNN round trip", len(rows) == 3 and all_finite and non_identical)


# ---------------------------------------------------------------------------
# 4. WAL two-writer behaviour
# ---------------------------------------------------------------------------


def check_wal_two_writers() -> None:
    section("4. WAL two-writer behaviour")
    tmp_dir = tempfile.mkdtemp(prefix="argus_wal_probe_")
    db_path = Path(tmp_dir) / "wal_probe.db"

    conn = sqlite3.connect(db_path, isolation_level=None)
    mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    print(f"journal_mode = {mode}")
    record("PRAGMA journal_mode reports 'wal'", mode.lower() == "wal")

    conn.execute(
        "CREATE TABLE wal_test (id INTEGER PRIMARY KEY AUTOINCREMENT, worker INTEGER, i INTEGER)"
    )
    conn.close()

    start = time.perf_counter()
    procs = [
        subprocess.Popen(
            [sys.executable, THIS_FILE, "--wal-worker", str(db_path), str(worker_id)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for worker_id in (1, 2)
    ]

    total_errors = 0
    nonzero_returncodes = []
    for worker_id, proc in zip((1, 2), procs):
        try:
            out, err = proc.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            print(f"worker {worker_id} TIMED OUT after 120s")
        print(f"--- worker {worker_id} stdout ---\n{out.strip()}")
        if err.strip():
            print(f"--- worker {worker_id} stderr ---\n{err.strip()}")
        if proc.returncode != 0:
            nonzero_returncodes.append((worker_id, proc.returncode))
        for line in out.splitlines():
            if line.startswith("WAL_WORKER_RESULT"):
                total_errors += int(line.rsplit("errors=", 1)[1])

    elapsed = time.perf_counter() - start

    verify_conn = sqlite3.connect(db_path)
    row_count = verify_conn.execute("SELECT COUNT(*) FROM wal_test").fetchone()[0]
    verify_conn.close()

    print(
        f"elapsed = {elapsed:.3f}s, row_count = {row_count}, total_errors = {total_errors}, "
        f"nonzero_returncodes = {nonzero_returncodes}"
    )
    record(
        "WAL two-writer: 400 rows landed, zero sqlite3.OperationalError",
        row_count == 400 and total_errors == 0 and not nonzero_returncodes,
    )


# ---------------------------------------------------------------------------
# 5. Ollama model co-residency
# ---------------------------------------------------------------------------


def _ollama_ps() -> None:
    try:
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=30)
        print(result.stdout.strip() or "(empty stdout)")
        if result.stderr.strip():
            print(f"stderr: {result.stderr.strip()}")
    except Exception:
        print(traceback.format_exc())


def check_ollama_coresidency() -> None:
    section("5. Ollama model co-residency")
    try:
        from argus import config
        from ollama import Client
    except Exception:
        print(traceback.format_exc())
        record("ollama co-residency probe completed", False)
        return

    try:
        planner_model = config.model_for_role("planner")
        embed_model = config.load_config().get("models", {}).get("embed")
        host = config.ollama_options().get("host")
    except Exception:
        print(traceback.format_exc())
        record("ollama co-residency probe completed", False)
        return

    print(f"planner model (from config) = {planner_model}")
    print(f"embed model (from config)   = {embed_model}")
    if not embed_model:
        print("config has no models.embed entry")
        record("ollama co-residency probe completed", False)
        return

    client = Client(host=host, timeout=120)

    def _chat_call(label: str):
        start = time.perf_counter()
        try:
            resp = client.chat(
                model=planner_model,
                messages=[{"role": "user", "content": "Reply with exactly one word: ok"}],
                think=False,
                keep_alive="30m",
                options={"temperature": 0},
            )
        except Exception:
            print(traceback.format_exc())
            return None, time.perf_counter() - start
        elapsed = time.perf_counter() - start
        load_ns = getattr(resp, "load_duration", None)
        load_s = f"{load_ns / 1_000_000_000:.3f}s" if load_ns is not None else "unavailable"
        print(f"{label}: wall_clock={elapsed:.3f}s load_duration={load_s} (raw ns={load_ns})")
        return resp, elapsed

    print("\n-- call 1: planner chat --")
    resp1, _ = _chat_call("call1 (planner chat)")
    print("ollama ps after call 1:")
    _ollama_ps()

    print("\n-- call 2: embedding call --")
    start = time.perf_counter()
    try:
        embed_resp = client.embed(model=embed_model, input="probe text for embedding")
    except Exception:
        print(traceback.format_exc())
        embed_resp = None
    print(f"call2 (embed): wall_clock={time.perf_counter() - start:.3f}s")
    print("ollama ps after call 2:")
    _ollama_ps()

    print("\n-- call 3: planner chat again --")
    resp3, _ = _chat_call("call3 (planner chat)")
    print("ollama ps after call 3:")
    _ollama_ps()

    record("call 1 (planner chat) succeeded", resp1 is not None)
    record("call 2 (embedding) succeeded", embed_resp is not None)
    record("call 3 (planner chat) succeeded", resp3 is not None)

    if resp3 is None:
        print("\nVERDICT: cannot be determined - call 3 failed")
    else:
        load3_ns = getattr(resp3, "load_duration", None)
        if load3_ns is None:
            verdict = "UNKNOWN (call 3 response carried no load_duration field)"
        elif load3_ns > 1_000_000_000:
            verdict = "EVICTS"
        else:
            verdict = "CO-RESIDENT"
        print(f"\nVERDICT: {verdict} (call3 load_duration = {load3_ns} ns)")
        record(
            f"planner+embed co-resident (call 3 paid no cold load) -> {verdict}",
            verdict == "CO-RESIDENT",
        )

    if embed_resp is None:
        record("embedding vector length == 768", False)
        return
    embeddings = getattr(embed_resp, "embeddings", None)
    vec_len = len(embeddings[0]) if embeddings else None
    print(f"embedding vector length = {vec_len}")
    record("embedding vector length == 768", vec_len == 768)


# ---------------------------------------------------------------------------
# 6. Node.js
# ---------------------------------------------------------------------------


def _version_cmd(cmd: "list[str]"):
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, shell=(os.name == "nt")
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except FileNotFoundError as exc:
        return None, str(exc), None
    except subprocess.TimeoutExpired as exc:
        return None, f"timed out after 30s: {exc}", None
    except Exception:
        return None, traceback.format_exc(), None


def check_node() -> None:
    section("6. Node.js")

    node_out, node_err, _ = _version_cmd(["node", "--version"])
    if node_out is None:
        print(f"node --version: NOT FOUND ({node_err})")
        record("node.js present", False)
        record("node >= 22.19.0 (MCP Inspector v2 requirement)", False)
    else:
        print(f"node --version = {node_out}")
        record("node.js present", True)
        try:
            ver = tuple(int(p) for p in node_out.lstrip("v").split("."))
            record("node >= 22.19.0 (MCP Inspector v2 requirement)", ver >= (22, 19, 0))
        except Exception:
            print(traceback.format_exc())
            record("node >= 22.19.0 (MCP Inspector v2 requirement)", False)

    npx_out, npx_err, _ = _version_cmd(["npx", "--version"])
    if npx_out is None:
        print(f"npx --version: NOT FOUND ({npx_err})")
        record("npx present", False)
    else:
        print(f"npx --version = {npx_out}")
        record("npx present", True)


# ---------------------------------------------------------------------------
# 7. Repo state
# ---------------------------------------------------------------------------


def check_repo_state() -> None:
    section("7. Repo state")
    root = Path(__file__).resolve().parents[1]

    def _run(cmd: "list[str]") -> bool:
        try:
            result = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=30)
            print(result.stdout.strip())
            if result.stderr.strip():
                print(f"stderr: {result.stderr.strip()}")
            return result.returncode == 0
        except Exception:
            print(traceback.format_exc())
            return False

    print("-- git log --oneline -5 --")
    record("git log succeeded", _run(["git", "log", "--oneline", "-5"]))

    print("\n-- git status --short --")
    record("git status succeeded", _run(["git", "status", "--short"]))

    notes_dir = root / "notes"
    prompts_dir = root / "src" / "argus" / "orchestrator" / "prompts"
    migrations_dir = root / "src" / "argus" / "storage" / "migrations"

    print(f"\nnotes/ exists      = {notes_dir.exists()} ({notes_dir})")
    print(f"prompts/ exists    = {prompts_dir.exists()} ({prompts_dir})")
    print(f"migrations/ exists = {migrations_dir.exists()} ({migrations_dir})")

    record("notes/ exists", notes_dir.exists())
    record("prompts/ exists (src/argus/orchestrator/prompts)", prompts_dir.exists())
    record("migrations/ exists (src/argus/storage/migrations)", migrations_dir.exists())


# ---------------------------------------------------------------------------


def main() -> None:
    check_python_sqlite()
    conn = check_sqlite_vec_loads()
    check_768_round_trip(conn)
    if conn is not None:
        conn.close()
    check_wal_two_writers()
    check_ollama_coresidency()
    check_node()
    check_repo_state()

    print("\n=== SUMMARY ===")
    for name, ok in RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")


if __name__ == "__main__":
    main()
