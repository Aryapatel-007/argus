"""Tests for the Phase C audit fixes (notes/audit.md).

Each guarantee has a negative control: the same check run against a
deliberately broken version must fail, so the test cannot pass vacuously.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest
import sqlite3

from argus import embeddings
from argus.orchestrator import llm, loop, tools
from argus.orchestrator.context import RunContext
from argus.orchestrator.state import State
from argus.servers import memory
from argus.storage import db, lease
from argus.storage.migrate import apply_migrations

from tests.test_loop import MockOllama

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "argus.db"
    apply_migrations(path)
    return path


@pytest.fixture
def mock_ollama():
    fake = MockOllama()
    with mock.patch.object(llm, "_invoke", fake):
        yield fake


# ---------------------------------------------------------------------------
# C1: the gate runs BEFORE the tool
# ---------------------------------------------------------------------------


def _planner_for(tool: str, flag: bool):
    def plan(ctx: RunContext, conn: Any = None) -> Dict[str, Any]:
        if ctx.facts_staged or ctx.current_task is not None:
            return {"done": True, "parse_ok": True, "tasks": []}
        return {"done": False, "parse_ok": True, "tasks": [{
            "description": f"call {tool}", "tool": tool,
            "action": {"tool": tool, "args": {"to": "x@example.invalid"},
                       "externally_visible": flag},
        }]}
    return plan


def _approvals(db_path):
    with db.connect(db_path) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM approvals")]


def _run_external(db_path, monkeypatch, *, planner_flag: bool):
    calls = []
    monkeypatch.setitem(tools.REGISTRY, "send_email",
                        lambda args: calls.append(args) or {"content": "sent"})
    monkeypatch.setattr(tools, "SIDE_EFFECT_TOOLS", frozenset({"send_email"}))
    ctx = loop.new_run("t1")
    loop.run_to_completion(ctx, db_path=db_path,
                           stubs=loop.Stubs(plan=_planner_for("send_email", planner_flag)))
    return ctx, calls


@pytest.mark.parametrize("planner_flag", [False, True])
def test_external_tool_never_runs_and_queues_one_approval(db_path, mock_ollama, monkeypatch, planner_flag):
    ctx, calls = _run_external(db_path, monkeypatch, planner_flag=planner_flag)
    assert calls == [], "external tool executed without approval"
    rows = _approvals(db_path)
    assert len(rows) == 1 and rows[0]["status"] == "pending"
    assert rows[0]["action_kind"] == "send_email"
    assert ctx.state is State.FAILED


def test_negative_control_without_the_gate_the_tool_runs(db_path, mock_ollama, monkeypatch):
    monkeypatch.setattr(tools, "is_external", lambda tool, flag: False)
    _, calls = _run_external(db_path, monkeypatch, planner_flag=False)
    assert len(calls) >= 1


def test_gate_step_replayed_after_crash_does_not_duplicate_the_approval(db_path, mock_ollama, monkeypatch):
    monkeypatch.setitem(tools.REGISTRY, "send_email", lambda args: {"content": "sent"})
    monkeypatch.setattr(tools, "SIDE_EFFECT_TOOLS", frozenset({"send_email"}))
    stubs = loop.Stubs(plan=_planner_for("send_email", False))
    ctx = loop.new_run("t1")
    loop.run_to_completion(ctx, db_path=db_path, stubs=stubs, max_steps=2)  # INIT, PLAN
    assert ctx.state is State.GATE
    # Replay the GATE handler's ops twice, as a crashed-and-resumed step would.
    with db.connect(db_path) as conn:
        _, ops = loop._on_gate(RunContext.from_dict(ctx.to_dict()), stubs, conn)
        for _ in range(2):
            with db.atomic(conn):
                for sql, params in ops[:1]:
                    conn.execute(sql, params)
    assert len(_approvals(db_path)) == 1


def test_planner_flag_alone_gates_a_tool_with_no_static_flag(db_path, mock_ollama, monkeypatch):
    """Review L-3: SIDE_EFFECT_TOOLS empty; only the planner says external."""
    calls = []
    monkeypatch.setitem(tools.REGISTRY, "post_comment",
                        lambda args: calls.append(args) or {"content": "posted"})
    monkeypatch.setattr(tools, "SIDE_EFFECT_TOOLS", frozenset())
    ctx = loop.new_run("t1")
    loop.run_to_completion(ctx, db_path=db_path,
                           stubs=loop.Stubs(plan=_planner_for("post_comment", True)))
    assert calls == []
    assert [r["status"] for r in _approvals(db_path)] == ["pending"]


def test_observe_refuses_an_external_tool_even_if_act_swapped_it_in(db_path, mock_ollama, monkeypatch):
    """Review M-2: the place the tool runs refuses external tools itself."""
    calls = []
    monkeypatch.setitem(tools.REGISTRY, "send_email",
                        lambda args: calls.append(args) or {"content": "sent"})
    monkeypatch.setattr(tools, "SIDE_EFFECT_TOOLS", frozenset({"send_email"}))

    def sneaky_act(ctx, conn=None):
        return {"tool": "send_email", "args": {}, "externally_visible": False}

    ctx = loop.new_run("t1")
    loop.run_to_completion(ctx, db_path=db_path, stubs=loop.Stubs(
        plan=_planner_for("stub_fetch_ok", False), act=sneaky_act))
    assert calls == []


def test_malformed_tool_name_fails_closed():
    assert tools.is_external(["not", "hashable"], False) is True
    assert tools.run_tool(["x"], {})["ok"] is False


def test_run_tool_never_raises_on_unexpected_errors(monkeypatch):
    def boom(args):
        raise ValueError("bad page")
    monkeypatch.setitem(tools.REGISTRY, "boom", boom)
    out = tools.run_tool("boom", {})
    assert out["ok"] is False and "ValueError" in out["error"]


# ---------------------------------------------------------------------------
# memory server helpers: deterministic fake embeddings, no Ollama needed
# ---------------------------------------------------------------------------


def _unit(seed: str):
    import hashlib, math
    raw = hashlib.sha256(seed.encode()).digest() * 24
    vec = [(b - 127.5) for b in raw[:768]]
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


@pytest.fixture
def fake_embed(monkeypatch):
    calls = []

    async def embed(text, *, run_id, conn=None, task_id=None, step_index=None):
        calls.append(text)
        return _unit(text)

    monkeypatch.setattr(embeddings, "embed", embed)
    return calls


def _seed_run(db_path, run_id="run-1"):
    with db.connect(db_path) as conn, db.atomic(conn):
        conn.execute("INSERT INTO runs (run_id, target_id, status) VALUES (?, 't1', 'running')", (run_id,))


def _current_without_vector(db_path):
    with db.connect(db_path) as conn:
        return conn.execute(
            "SELECT f.fact_id FROM facts f LEFT JOIN facts_vec v ON v.fact_id = f.fact_id "
            "WHERE f.is_current = 1 AND v.fact_id IS NULL"
        ).fetchall()


# ---------------------------------------------------------------------------
# H1: never a current fact without its vector
# ---------------------------------------------------------------------------


def test_stale_advisory_read_still_embeds_the_new_version(db_path, fake_embed, monkeypatch):
    _seed_run(db_path)
    server = memory.MemoryServer(db_path, "run-1")
    asyncio.run(server.remember_fact("t1", "fees", "Fees are AUD 40,000."))

    new_content = "Fees are AUD 45,000."
    real_current_row = memory.MemoryServer._current_row
    calls = {"n": 0}

    def stale_first(conn, target_id, entity_key):
        calls["n"] += 1
        if calls["n"] == 1:  # the advisory read: pretend it already matches
            return {"fact_id": -1, "content_hash": memory.content_hash(new_content, None),
                    "version": 9, "first_seen_run_id": "run-1", "first_seen_at": "x"}
        return real_current_row(conn, target_id, entity_key)

    monkeypatch.setattr(memory.MemoryServer, "_current_row", staticmethod(stale_first))
    result = asyncio.run(server.remember_fact("t1", "fees", new_content))

    assert result.change_kind == "modified" and result.version == 2
    assert _current_without_vector(db_path) == []
    assert new_content in fake_embed


def test_negative_control_vectorless_current_fact_is_detected(db_path):
    """The invariant query above must be able to see the H1 damage."""
    _seed_run(db_path)
    with db.connect(db_path) as conn, db.atomic(conn):
        conn.execute(
            "INSERT INTO facts (target_id, entity_key, content, content_hash, change_kind, "
            "first_seen_run_id, last_seen_run_id, first_seen_at, last_seen_at) "
            "VALUES ('t1','fees','x','h','added','run-1','run-1','a','a')")
    assert len(_current_without_vector(db_path)) == 1


# ---------------------------------------------------------------------------
# H3: idempotent remember_fact
# ---------------------------------------------------------------------------


def _fact_rows(db_path):
    with db.connect(db_path) as conn:
        return conn.execute("SELECT fact_id, version, content FROM facts ORDER BY fact_id").fetchall()


def test_same_key_replays_the_first_result_and_writes_nothing(db_path, fake_embed):
    _seed_run(db_path)
    server = memory.MemoryServer(db_path, "run-1")
    first = asyncio.run(server.remember_fact("t1", "fees", "Fees: AUD 45,000.", idempotency_key="k1"))
    again = asyncio.run(server.remember_fact("t1", "fees", "Fees: AUD 45,000.", idempotency_key="k1"))
    drifted = asyncio.run(server.remember_fact("t1", "fees", "Fees are 45k AUD.", idempotency_key="k1"))
    assert first.change_kind == "added"
    assert again.model_dump_json() == first.model_dump_json()
    assert drifted.model_dump_json() == first.model_dump_json()
    assert len(_fact_rows(db_path)) == 1


def test_reusing_a_key_for_a_different_fact_raises(db_path, fake_embed):
    """Review M-5."""
    _seed_run(db_path)
    server = memory.MemoryServer(db_path, "run-1")
    asyncio.run(server.remember_fact("t1", "fees", "Fees: AUD 45,000.", idempotency_key="k9"))
    with pytest.raises(memory.IdempotencyKeyReuseError):
        asyncio.run(server.remember_fact("t1", "intake", "July.", idempotency_key="k9"))


def test_server_refuses_a_database_with_002_but_not_004(tmp_path):
    """Review L-5: 004 (receipts) is required on its own, not only via 002."""
    from argus.storage import migrate as migrate_mod
    import shutil
    partial_dir = tmp_path / "migs"
    partial_dir.mkdir()
    for f in migrate_mod.MIGRATIONS_DIR.glob("*.sql"):
        if f.name < "004":
            shutil.copy(f, partial_dir / f.name)
    path = tmp_path / "partial.db"
    apply_migrations(path, partial_dir)
    with pytest.raises(SystemExit) as caught:
        memory.assert_migrated(path)
    assert "004_remember_receipts.sql" in str(caught.value)


def test_negative_control_without_a_key_a_retry_misreports(db_path, fake_embed):
    _seed_run(db_path)
    server = memory.MemoryServer(db_path, "run-1")
    first = asyncio.run(server.remember_fact("t1", "fees", "Fees: AUD 45,000."))
    retry = asyncio.run(server.remember_fact("t1", "fees", "Fees: AUD 45,000."))
    drifted = asyncio.run(server.remember_fact("t1", "fees", "Fees are 45k AUD."))
    assert first.change_kind == "added"
    assert retry.change_kind == "unchanged"     # the misreport H3 describes
    assert drifted.change_kind == "modified"    # the phantom version


def test_importance_is_stored_and_range_checked(db_path, fake_embed):
    _seed_run(db_path)
    server = memory.MemoryServer(db_path, "run-1")
    asyncio.run(server.remember_fact("t1", "fees", "Fees: AUD 45,000.", importance=0.8))
    with db.connect(db_path) as conn:
        assert conn.execute("SELECT importance FROM facts").fetchone()[0] == 0.8
    with pytest.raises(ValueError):
        asyncio.run(server.remember_fact("t1", "x", "y", importance=1.5))


# ---------------------------------------------------------------------------
# H2: one process per run
# ---------------------------------------------------------------------------


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _start_run(db_path, owner):
    ctx = loop.new_run("t1")
    loop.run_to_completion(ctx, db_path=db_path, stubs=loop.Stubs(plan=_planner_for("stub_fetch_ok", False)),
                           max_steps=2, lease_owner=owner)
    return ctx


def test_live_owner_blocks_a_second_process(db_path, mock_ollama):
    import socket
    owner_a = f"{socket.gethostname()}:{os.getpid()}:aaaa"
    ctx = _start_run(db_path, owner_a)
    with db.connect(db_path) as conn:
        assert lease.acquire(conn, ctx.run_id, "otherhost:1:bbbb") is False
        before = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
    # B tries to step anyway: nothing commits.
    clone = loop.load_latest_checkpoint(ctx.run_id, db_path=db_path)
    with db.connect(db_path) as conn:
        task_status = conn.execute("SELECT status FROM tasks").fetchone()[0]
    assert clone.state is State.ACT and task_status == "pending"
    with pytest.raises(lease.LeaseLost):
        loop.step(clone, db_path=db_path, lease_owner="otherhost:1:bbbb")
    with db.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == before
        # The ACT step's own write (tasks.status -> running) ran before the
        # lease check failed; it must have rolled back with the step. This is
        # what proves the step's ops and its checkpoint share one transaction.
        assert conn.execute("SELECT status FROM tasks").fetchone()[0] == "pending"


def test_dead_owner_is_taken_over_immediately(db_path, mock_ollama):
    import socket
    dead = f"{socket.gethostname()}:{_dead_pid()}:dddd"
    ctx = _start_run(db_path, dead)
    with db.connect(db_path) as conn:
        assert lease.acquire(conn, ctx.run_id, "me:1:eeee") is True
        assert lease.current_owner(conn, ctx.run_id) == "me:1:eeee"


def test_stale_heartbeat_is_taken_over(db_path, mock_ollama):
    ctx = _start_run(db_path, "elsewhere:123:ffff")  # other host: liveness unknown
    with db.connect(db_path) as conn:
        assert lease.acquire(conn, ctx.run_id, "me:1:gggg") is False
        with db.atomic(conn):
            conn.execute("UPDATE run_leases SET heartbeat_at = '2000-01-01T00:00:00.000Z'")
        assert lease.acquire(conn, ctx.run_id, "me:1:gggg") is True


def test_unique_index_refuses_a_forked_step_even_without_leases(db_path, mock_ollama):
    ctx = _start_run(db_path, None)
    a = loop.load_latest_checkpoint(ctx.run_id, db_path=db_path)
    b = loop.load_latest_checkpoint(ctx.run_id, db_path=db_path)
    loop.step(a, db_path=db_path)
    with pytest.raises(sqlite3.IntegrityError):
        loop.step(b, db_path=db_path)


def test_negative_control_without_the_index_a_fork_commits(db_path, mock_ollama):
    with db.connect(db_path) as conn:
        conn.execute("DROP INDEX idx_checkpoints_run_step_unique")
    ctx = _start_run(db_path, None)
    a = loop.load_latest_checkpoint(ctx.run_id, db_path=db_path)
    b = loop.load_latest_checkpoint(ctx.run_id, db_path=db_path)
    loop.step(a, db_path=db_path)
    loop.step(b, db_path=db_path)
    with db.connect(db_path) as conn:
        dupes = conn.execute(
            "SELECT step_index FROM checkpoints GROUP BY run_id, step_index HAVING COUNT(*) > 1"
        ).fetchall()
    assert dupes


def test_pid_alive_reports_self_alive_and_exited_child_dead():
    assert lease.pid_alive(os.getpid())
    assert not lease.pid_alive(_dead_pid())


@pytest.mark.skipif(sys.platform != "win32", reason="Windows access-denied path")
def test_pid_alive_treats_access_denied_as_alive():
    """Review M-1. PID 4 is the Windows System process: it exists, and a
    normal user cannot open it. Access denied must read as alive."""
    assert lease.pid_alive(4)


def test_live_same_host_owner_keeps_its_lease_even_when_stale(db_path, mock_ollama):
    """Review H-2: a slow step (two 120s planner attempts) outlasts the stale
    window. On the same host the pid is authoritative."""
    import socket
    live = f"{socket.gethostname()}:{os.getpid()}:hhhh"
    ctx = _start_run(db_path, live)
    with db.connect(db_path) as conn:
        with db.atomic(conn):
            conn.execute("UPDATE run_leases SET heartbeat_at = '2000-01-01T00:00:00.000Z'")
        assert lease.acquire(conn, ctx.run_id, "x:1:iiii") is False
        assert lease.current_owner(conn, ctx.run_id) == live


# ---------------------------------------------------------------------------
# M1, M4, M5
# ---------------------------------------------------------------------------


def test_critic_failures_do_not_spend_the_planner_budget(db_path):
    ctx = loop.new_run("t1")
    with mock.patch.object(llm, "_invoke", side_effect=ConnectionError("down")):
        llm.call_ollama("critic", "critic.md", {"action": "{}", "result": "{}"}, ctx=ctx)
        llm.call_ollama("critic", "critic.md", {"action": "{}", "result": "{}"}, ctx=ctx)
        assert ctx.parse_retries == 0
        llm.call_ollama("planner", "planner.md", {"goal": "g", "target_id": "t",
                                                  "tool_catalog": "", "max_tasks": 1}, ctx=ctx)
    assert ctx.parse_retries == 1


def _env_without_pythonpath():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env


@pytest.mark.parametrize("module", ["argus.servers.memory", "argus.orchestrator.runner"])
def test_python_dash_m_works_without_pythonpath(module):
    out = subprocess.run([sys.executable, "-m", module, "--help"], cwd=REPO,
                         env=_env_without_pythonpath(), capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr


@pytest.mark.parametrize("round_", range(4))
def test_concurrent_migrators_on_a_fresh_file_all_succeed(tmp_path, round_):
    """Review H-1: the race was on the very first connect (WAL pragma), and
    reproduced about 1 run in 4-10. Four rounds of three processes each."""
    target = tmp_path / "fresh.db"
    code = ("import sys; from argus.storage.migrate import apply_migrations; "
            "apply_migrations(sys.argv[1])")
    procs = [subprocess.Popen([sys.executable, "-c", code, str(target)], cwd=REPO,
                              env=_env_without_pythonpath(), stderr=subprocess.PIPE, text=True)
             for _ in range(3)]
    results = [(p.wait(timeout=180), p.stderr.read()) for p in procs]
    assert all(rc == 0 for rc, _ in results), results
    with db.connect(target) as conn:
        names = [r[0] for r in conn.execute("SELECT filename FROM schema_migrations")]
    assert len(names) == len(set(names))
