"""Dashboard (FastAPI) and approval CLI on top of a real fake-model run."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from argus import app as argus_app
from argus import cli, observe
from argus.dashboard import create_app

from tests.test_graph_engine import PAGE_V1, env, fake_embed, q  # noqa: F401
from tests.test_graph_review_fixes import run


@pytest.fixture
def waiting(env):
    db_path, outbox = env
    _, out = run(db_path, outbox, PAGE_V1)
    assert out.status == "waiting_approval"
    return db_path, outbox, out


def client(db_path, token="t0k"):
    # "testserver" is TestClient's Host header; production never allows it.
    return TestClient(create_app(db_path, csrf_token=token, extra_hosts=("testserver",)))


def test_pages_render_from_a_real_run(waiting):
    db_path, _, out = waiting
    c = client(db_path)
    assert out.run_id in c.get("/").text
    page = c.get(f"/runs/{out.run_id}").text
    assert "extract" in page and "swap" in page and "45,000" in page
    node = q(db_path, "SELECT node_id FROM graph_nodes WHERE node_type='extract'")[0]["node_id"]
    assert "Utility" in c.get(f"/nodes/{node}").text
    assert "Approve" in c.get("/approvals").text
    assert c.get("/health").status_code == 200
    assert c.get("/runs/nope").status_code == 404


def test_approve_requires_the_csrf_token_and_a_local_host(waiting):
    db_path, _, _ = waiting
    aid = q(db_path, "SELECT approval_id FROM approvals")[0]["approval_id"]
    c = client(db_path)
    assert c.post(f"/approvals/{aid}", data={"decision": "approve", "csrf": "wrong"}).status_code == 403
    assert c.post(f"/approvals/{aid}", data={"decision": "approve", "csrf": "t0k"},
                  headers={"host": "evil.example"}).status_code == 403
    assert q(db_path, "SELECT status FROM approvals")[0]["status"] == "pending"
    r = c.post(f"/approvals/{aid}", data={"decision": "approve", "csrf": "t0k"}, follow_redirects=False)
    assert r.status_code == 303
    assert q(db_path, "SELECT status, decided_by FROM approvals")[0] == {"status": "approved", "decided_by": "dashboard"}
    again = c.post(f"/approvals/{aid}", data={"decision": "reject", "csrf": "t0k"})
    assert again.status_code == 409, "a decided approval cannot be decided again"


def test_cli_lists_approves_and_the_run_completes_once(waiting, capsys):
    db_path, outbox, out = waiting
    assert cli.main(["--db", str(db_path), "approvals", "list"]) == 0
    listing = capsys.readouterr().out
    aid = q(db_path, "SELECT approval_id FROM approvals")[0]["approval_id"]
    assert f"#{aid}" in listing and "45,000" in listing
    assert cli.main(["--db", str(db_path), "approvals", "approve", str(aid)]) == 0
    assert argus_app.decided_waiting_runs(db_path) == [out.run_id]
    outs = asyncio.run(argus_app.resume_decided(argus_app.Options(db_path=db_path, memory="direct",
                                                                  pages={}, fake_backend=True, outbox=outbox)))
    assert [o.status for o in outs] == ["done"]
    assert len(list(outbox.iterdir())) == 1
    assert cli.main(["--db", str(db_path), "approvals", "approve", str(aid)]) == 2  # not pending any more


def test_cli_trace_and_why_explain_from_the_database(waiting, capsys):
    db_path, _, out = waiting
    cli.main(["--db", str(db_path), "trace", out.run_id])
    text = capsys.readouterr().out
    assert "critique" in text and "swaps=" in text
    node = q(db_path, "SELECT node_id FROM graph_nodes WHERE node_type='critique'")[0]["node_id"]
    cli.main(["--db", str(db_path), "why", node])
    assert "utility=" in capsys.readouterr().out


def test_swaps_counts_model_changes():
    assert observe.swaps(["a", "a", "b", "b", "a"]) == 2
    assert observe.swaps([]) == 0
