"""Tests for the Phase F review findings."""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from argus import app as argus_app
from argus import scheduler as sched_mod
from argus.dashboard import create_app
from argus.eval.fixture_server import FixtureServer
from argus.graph import handlers
from argus.servers.browser import BrowserBackend, request_allowed
from argus.storage import db

from tests.test_graph_engine import PAGE_V1, TARGET, URL, env, fake_embed, q  # noqa: F401
from tests.test_graph_review_fixes import run


# -- H1: one failing run never blocks recovery of the others ---------------


def test_recover_continues_past_a_failing_run(env, monkeypatch):
    db_path, _ = env
    s = sched_mod.Scheduler(argus_app.Options(db_path=db_path, memory="direct", pages={}, fake_backend=True))
    monkeypatch.setattr(argus_app, "decided_waiting_runs", lambda p: ["approved-run"])
    monkeypatch.setattr(sched_mod, "interrupted_runs", lambda p: ["broken-run", "approved-run"])
    resumed = []

    async def resume(self, run_id):
        if run_id == "broken-run":
            raise RuntimeError("bad stored data")
        resumed.append(run_id)
        return []

    monkeypatch.setattr(argus_app.Session, "resume", resume)
    for _ in range(sched_mod.MAX_RECOVER_ATTEMPTS + 2):
        asyncio.run(s.recover())
    assert resumed[0] == "approved-run", "decided approvals go first"
    assert s.failures["broken-run"] == sched_mod.MAX_RECOVER_ATTEMPTS, "parked after the cap"
    assert len(resumed) == sched_mod.MAX_RECOVER_ATTEMPTS + 2


# -- M1: a memory server that cannot start does not strand the lease -------


def test_failed_memory_start_releases_the_lease(env, monkeypatch):
    db_path, _ = env

    async def boom(self, run_id):
        raise OSError("memory server failed to spawn")

    monkeypatch.setattr("argus.graph.mcp_clients.RunMemory.open", boom)
    opts = argus_app.Options(db_path=db_path, memory="direct", pages={URL: PAGE_V1}, fake_backend=True)

    async def go():
        async with argus_app.Session(opts) as s:
            await s.run_target(TARGET)

    with pytest.raises(OSError):
        asyncio.run(go())
    assert q(db_path, "SELECT COUNT(*) AS n FROM run_leases")[0]["n"] == 0
    run_id = q(db_path, "SELECT run_id FROM runs")[0]["run_id"]
    assert run_id in sched_mod.interrupted_runs(db_path), "recovery must be able to pick it up"


# -- M2: network boundary -----------------------------------------------------


@pytest.mark.parametrize("url,page_host,ok", [
    ("http://127.0.0.1:8765/a", "127.0.0.1:8765", True),       # the page's own (local) origin
    ("http://127.0.0.1:8787/approvals", "127.0.0.1:8765", False),  # the dashboard
    ("http://127.0.0.1:11434/api/ps", "uni.edu:443", False),   # Ollama from a public page
    ("http://192.168.1.1/", "uni.edu:443", False),
    ("http://169.254.169.254/latest", "uni.edu:443", False),
    ("http://localhost/x", "uni.edu:443", False),
    ("https://cdn.example.com/app.js", "uni.edu:443", True),
])
def test_request_filter(url, page_host, ok):
    assert request_allowed(url, page_host) is ok


@pytest.mark.browser
def test_redirect_to_another_origin_is_refused_and_internal_requests_are_blocked():
    with FixtureServer({"/secret": "<html><body>internal secret</body></html>"}) as other:
        page = (f'<html><body><p>Fee AUD 45,000.</p><img src="{other.url("/secret")}">'
                f'<script>fetch("{other.url("/secret")}").catch(() => 0)</script></body></html>')
        redirect = f'<html><head><meta http-equiv="refresh" content="0;url={other.url("/secret")}"></head></html>'
        with FixtureServer({"/p": page, "/r": redirect}) as site:
            async def go():
                b = BrowserBackend(throttle_s=0, idle_timeout_s=2)
                try:
                    return await b.open_page(site.url("/p")), await b.open_page(site.url("/r"))
                finally:
                    await b.close()
            good, redirected = asyncio.run(go())
        hits = other.hits["/secret"]
    assert good.ok and "45,000" in good.text
    assert hits == 0, "a page must not reach another internal origin"
    assert not redirected.ok and ("another origin" in (redirected.error or "") or "secret" not in redirected.text)


def test_fetch_handler_refuses_an_off_site_final_url():
    class B:
        async def open_page(self, url):
            return {"ok": True, "text": "x", "final_url": "https://evil.example/landing"}
    ex = SimpleNamespace(inputs={}, params={"url": URL}, target=TARGET, services=SimpleNamespace(browser=B()))
    r = asyncio.run(handlers._fetch(ex))
    assert r.status == "failed" and "off-site" in r.error


# -- M3, M4: bounded reads, relaunch ----------------------------------------


@pytest.mark.browser
def test_busy_looping_page_degrades_within_the_bound():
    busy = "<html><body><p>Fee AUD 45,000.</p><script>setTimeout(() => { while (true) {} }, 50)</script></body></html>"
    with FixtureServer({"/busy": busy}) as site:
        async def go():
            b = BrowserBackend(throttle_s=0, idle_timeout_s=1, read_timeout_s=3)
            try:
                t0 = time.monotonic()
                r = await b.open_page(site.url("/busy"))
                return r, time.monotonic() - t0, b.open_contexts
            finally:
                await b.close()
        r, elapsed, contexts = asyncio.run(go())
    assert elapsed < 30
    assert contexts == 0
    assert r.ok is False or "45,000" in r.text  # either degraded or read before the loop started


@pytest.mark.browser
def test_crashed_chromium_is_relaunched():
    with FixtureServer({"/p": "<html><body><p>hello</p></body></html>"}) as site:
        async def go():
            b = BrowserBackend(throttle_s=0)
            try:
                first = await b.open_page(site.url("/p"))
                await b._browser.close()  # Chromium goes away under the server
                second = await b.open_page(site.url("/p"))
                return first, second
            finally:
                await b.close()
        first, second = asyncio.run(go())
    assert first.ok and second.ok


# -- dashboard ------------------------------------------------------------------


def test_dashboard_escapes_page_derived_text_and_hides_docs(env):
    db_path, outbox = env
    _, out = run(db_path, outbox, PAGE_V1)
    with db.connect(db_path) as conn, db.atomic(conn):
        conn.execute("UPDATE approvals SET proposed_action = ?", (json.dumps(
            {"subject": "<script>alert(1)</script>", "body": "<img src=x onerror=alert(2)>"}),))
    c = TestClient(create_app(db_path, csrf_token="t", extra_hosts=("testserver",)))
    html = c.get("/approvals").text
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html
    assert "<img src=x" not in html
    for path in ("/docs", "/openapi.json", "/redoc"):
        assert c.get(path).status_code == 404


def test_dashboard_refuses_foreign_hosts_on_reads_too(env):
    db_path, _ = env
    c = TestClient(create_app(db_path, csrf_token="t"))  # production host list only
    assert c.get("/", headers={"host": "evil.example"}).status_code == 403
    assert c.get("/", headers={"host": "127.0.0.1:8787"}).status_code == 200


def test_dashboard_reject_path_and_non_ascii_token(env):
    db_path, outbox = env
    run(db_path, outbox, PAGE_V1)
    aid = q(db_path, "SELECT approval_id FROM approvals")[0]["approval_id"]
    c = TestClient(create_app(db_path, csrf_token="t", extra_hosts=("testserver",)))
    assert c.post(f"/approvals/{aid}", data={"decision": "reject", "csrf": "tö"}).status_code == 403
    assert c.post(f"/approvals/{aid}", data={"decision": "reject", "csrf": "t"},
                  follow_redirects=False).status_code == 303
    assert q(db_path, "SELECT status FROM approvals")[0]["status"] == "rejected"


def test_child_server_environment_is_allowlisted(monkeypatch):
    from argus.graph import mcp_clients
    monkeypatch.setenv("SOME_API_TOKEN", "secret")
    monkeypatch.setenv("ARGUS_FAKE_EMBED", "1")
    env = mcp_clients._server_env()
    assert "SOME_API_TOKEN" not in env and env.get("ARGUS_FAKE_EMBED") == "1"
    assert any(k.upper() == "SYSTEMROOT" for k in env) or "SYSTEMROOT" not in {k.upper() for k in __import__("os").environ}
