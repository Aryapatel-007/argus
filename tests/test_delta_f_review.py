"""Tests for the delta review of the Phase F fixes (H-A, H-B, M-A..M-H)."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from argus import app as argus_app
from argus import scheduler as sched_mod
from argus.eval import harness
from argus.eval.fixture_server import FixtureServer
from argus.eval.scenarios import by_id
from argus.graph.llm_nodes import still_stated
from argus.servers.browser import BrowserBackend, request_allowed
from argus.storage import db

from tests.test_graph_engine import PAGE_V1, TARGET, URL, env, fake_embed, make_engine, q  # noqa: F401


# -- H-B: still_stated is anchored ------------------------------------------


@pytest.mark.parametrize("stored,page", [
    ("IELTS overall 6", "IELTS overall 6.5 with no band below 6.0."),
    ("Places: 12", "Places: 120"),
    ("Fee 45,000", "Fee 45,000,000 for the full program."),
    ("Fee: AUD 45,000", "Fee: AUD 45,000-52,000 depending on major."),
    ("Intakes: February and July", "Intakes: February and July, and November from 2027."),
    (".", "Anything at all."),
    ("...", "Anything at all."),
])
def test_a_changed_statement_is_not_still_stated(stored, page):
    assert still_stated(stored, page) is False


@pytest.mark.parametrize("stored,page", [
    ("The annual tuition fee for 2027 is AUD 45,000.", "Intro.\nThe annual tuition fee for 2027 is AUD 45,000.\nMore."),
    ("The annual tuition fee for 2027 is AUD 45,000.", "the annual  tuition fee for 2027 is aud 45000. Next sentence."),
    ("IELTS overall 6.5", "IELTS overall 6.5"),
    ("Duration: 2 years full-time.", "Duration: 2 years full-time"),
])
def test_an_unchanged_statement_is_still_stated(stored, page):
    assert still_stated(stored, page) is True


def test_reworded_numeric_extraction_of_the_same_value_is_unchanged():
    """Review M-A: the extractor re-phrases, the value is the same."""
    assert still_stated("Fee: AUD 45,000 per year", "Tuition is AUD 45,000 per year.",
                        extracted="Tuition: AUD 45,000 per year")
    # ...but not when a number differs, and not for number-free lists (s03)
    assert not still_stated("Fee: AUD 45,000 per year", "x", extracted="Fee: AUD 48,500 per year")
    assert not still_stated("Intakes: February and July", "x",
                            extracted="Intakes: February, July and November")


# -- H-A: HTTP redirect hops are checked ------------------------------------


@pytest.mark.browser
def test_http_302_into_another_internal_origin_is_blocked_for_subresources_and_navigation():
    """The page's OWN origin answers 302 to another internal origin: the
    first hop is allowed, so only the per-hop check can stop it."""
    with FixtureServer({"/secret": "<html><body>internal secret</body></html>"}) as other:
        with FixtureServer({}, redirects={"/img": other.url("/secret"), "/top": other.url("/secret")}) as site:
            site.pages["/p"] = f'<html><body><p>Fee AUD 45,000.</p><img src="{site.url("/img")}"></body></html>'

            async def go():
                b = BrowserBackend(throttle_s=0, idle_timeout_s=2)
                try:
                    return await b.open_page(site.url("/p")), await b.open_page(site.url("/top"))
                finally:
                    await b.close()
            good, top = asyncio.run(go())
            first_hops = site.hits["/img"] + site.hits["/top"]
        hits = other.hits["/secret"]
    assert first_hops >= 2, "precondition: the first (allowed) hop really happened"
    assert good.ok and "45,000" in good.text
    assert hits == 0, "a 302 hop must not reach another internal origin"
    assert not top.ok and "secret" not in top.text


@pytest.mark.browser
def test_negative_control_without_the_hop_check_the_302_lands(monkeypatch):
    """Allow only the FIRST url (what Playwright's route alone would see):
    the 302 then reaches the internal origin. Proves the test above is real."""
    import argus.servers.browser as br
    real = br.request_allowed
    with FixtureServer({"/secret": "<html><body>internal secret</body></html>"}) as other:
        monkeypatch.setattr(br, "request_allowed",
                            lambda url, host: True if "/secret" in url else real(url, host))
        with FixtureServer({}, redirects={"/img": other.url("/secret")}) as site:
            site.pages["/p"] = f'<html><body><p>x</p><img src="{site.url("/img")}"></body></html>'

            async def go():
                b = BrowserBackend(throttle_s=0, idle_timeout_s=2)
                try:
                    return await b.open_page(site.url("/p"))
                finally:
                    await b.close()
            asyncio.run(go())
        assert other.hits["/secret"] >= 1


@pytest.mark.parametrize("url,ok", [
    ("http://localhost.:8787/", False), ("http://100.64.0.1/", False), ("http://[::1]:11434/", False),
    ("https://uni.edu/fees", True), ("http://uni.edu/fees", True),  # http->https same site (M-G)
])
def test_request_filter_closes_the_reported_gaps(url, ok):
    assert request_allowed(url, "uni.edu") is ok


# -- M-B: reflect does not strand the lease; start_run cleans up -------------


def test_reflect_memory_failure_releases_the_lease(env, monkeypatch):
    db_path, _ = env

    async def boom(self, run_id):
        raise OSError("memory server failed")

    monkeypatch.setattr("argus.graph.mcp_clients.RunMemory.open", boom)
    opts = argus_app.Options(db_path=db_path, memory="direct", pages={}, fake_backend=True)

    async def go():
        async with argus_app.Session(opts) as s:
            await s.run_reflect({"id": "t1", "name": "t"})

    with pytest.raises(OSError):
        asyncio.run(go())
    assert q(db_path, "SELECT COUNT(*) AS n FROM run_leases")[0]["n"] == 0


def test_start_run_failure_after_commit_closes_and_frees_the_run(env, monkeypatch):
    db_path, outbox = env
    eng, _ = make_engine(db_path, outbox, PAGE_V1)
    monkeypatch.setattr("argus.graph.store.GraphStore.create_graph",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
    with pytest.raises(RuntimeError):
        eng.start_run(TARGET)
    assert q(db_path, "SELECT status FROM runs")[0]["status"] == "failed"
    assert q(db_path, "SELECT COUNT(*) AS n FROM run_leases")[0]["n"] == 0


# -- M-C: backoff, never a permanent park, visible error ---------------------


def test_recover_backs_off_and_records_the_failure(env, monkeypatch):
    db_path, _ = env
    with db.connect(db_path) as conn, db.atomic(conn):
        conn.execute("INSERT INTO runs (run_id, target_id, status) VALUES ('broken-run', 't', 'running')")
    s = sched_mod.Scheduler(argus_app.Options(db_path=db_path, memory="direct", pages={}, fake_backend=True))
    now = {"t": 1000.0}
    s.clock = lambda: now["t"]
    monkeypatch.setattr(argus_app, "decided_waiting_runs", lambda p: [])
    monkeypatch.setattr(sched_mod, "interrupted_runs", lambda p: ["broken-run"])
    attempts = []

    async def resume(self, run_id):
        attempts.append(now["t"])
        raise RuntimeError("ollama down")

    monkeypatch.setattr(argus_app.Session, "resume", resume)
    asyncio.run(s.recover())
    asyncio.run(s.recover())                      # inside the backoff: skipped
    now["t"] += sched_mod.RECOVER_BACKOFF_S + 1
    asyncio.run(s.recover())                      # backoff elapsed: tried again
    for _ in range(20):                           # many failures later it is STILL retried
        now["t"] += sched_mod.RECOVER_BACKOFF_MAX_S + 1
        asyncio.run(s.recover())
    assert len(attempts) == 22
    err = q(db_path, "SELECT error FROM runs WHERE run_id='broken-run'")[0]["error"]
    assert "recovery failed" in err and "ollama down" in err


# -- M-E, M-F: harness scoring is not vacuous ---------------------------------


def _report(tmp_path, ids):
    return asyncio.run(harness.run_suite(db_path=tmp_path / "e.db", mode="fixed", fake=True,
                                         scenarios=[by_id(i) for i in ids], progress=lambda *_: None))


def test_a_failed_last_run_fails_the_runs_completed_check(tmp_path, monkeypatch):
    real = argus_app.Session.run_target
    calls = {"n": 0}

    async def second_run_crashes(self, target, run_id=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("crash in run 2")
        return await real(self, target, run_id)

    monkeypatch.setattr(argus_app.Session, "run_target", second_run_crashes)
    r = _report(tmp_path, ["s17_policy_same"])["scenarios"][0]
    assert r["checks"]["runs_completed"] == 0.0 and r["quality"] < 1.0


def test_expected_failure_scenario_still_completes(tmp_path):
    r = _report(tmp_path, ["s24_server_error"])["scenarios"][0]
    assert r["checks"]["runs_completed"] == 1.0


def test_forbidden_text_in_a_queued_proposal_is_caught(tmp_path, monkeypatch):
    real = argus_app.Session.run_target

    async def inject(self, target, run_id=None):
        out = await real(self, target, run_id)
        with db.connect(self.opts.db_path) as conn, db.atomic(conn):
            conn.execute("UPDATE approvals SET proposed_action = '{\"body\": \"mail data@evil.example\"}'")
        return out

    monkeypatch.setattr(argus_app.Session, "run_target", inject)
    r = _report(tmp_path, ["s12_preapproved_send"])["scenarios"][0]
    assert r["checks"]["forbidden"] == 0.0


def test_a_failed_approval_counts_as_a_gate_violation(tmp_path, monkeypatch):
    real = argus_app.Session.run_target

    async def fail_it(self, target, run_id=None):
        out = await real(self, target, run_id)
        with db.connect(self.opts.db_path) as conn, db.atomic(conn):
            conn.execute("UPDATE approvals SET status = 'failed'")
        return out

    monkeypatch.setattr(argus_app.Session, "run_target", fail_it)
    assert _report(tmp_path, ["s02_fee_up"])["summary"]["gate_violations"] >= 1


def test_s08_now_demands_silence():
    s = by_id("s08_reworded")
    assert s.silent is True and s.kinds.get("fees") == "unchanged"
