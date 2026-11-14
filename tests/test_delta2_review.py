"""Tests for the second delta review (of a602fb2): H1, H2, M1-M4, L1-L3, L6."""

from __future__ import annotations

import asyncio
import ipaddress
import re
from types import SimpleNamespace

import pytest
from hypothesis import given, strategies as st

from argus import app as argus_app
from argus import scheduler as sched_mod
from argus.eval import harness
from argus.eval.fixture_server import FixtureServer
from argus.eval.scenarios import by_id, program
from argus.graph.llm_nodes import still_stated
from argus.servers import browser as br
from argus.servers.browser import BrowserBackend, request_allowed
from argus.storage import db

from tests.test_graph_engine import env, q  # noqa: F401


# -- H1: a re-worded numeric fact is unchanged only if it says the same thing --


@pytest.mark.parametrize("stored,extracted", [
    ("Fee AUD 45,000 for domestic students", "Fee AUD 45,000 for international students"),
    ("Tuition AUD 45,000 per year", "Tuition AUD 45,000 per semester"),
    ("Deposit AUD 500, refundable", "Deposit AUD 500, not refundable"),
    ("Scholarship of AUD 10,000 available", "Scholarship of AUD 10,000 no longer available"),
    ("IELTS overall 6.5, no band below 6.0", "IELTS overall 6.0, no band below 6.5"),
    ("Domestic AUD 45,000; international AUD 52,000", "Domestic AUD 52,000; international AUD 45,000"),
])
def test_reworded_extraction_with_a_changed_qualifier_or_order_is_a_change(stored, extracted):
    assert still_stated(stored, "unrelated page text", extracted=extracted) is False


@pytest.mark.parametrize("stored,extracted", [
    ("Fee: AUD 45,000 per year", "Tuition: AUD 45,000 per year"),
    ("The annual tuition fee is AUD 45,000", "Annual fee: AUD 45,000"),
])
def test_reworded_extraction_with_only_label_words_changed_is_unchanged(stored, extracted):
    assert still_stated(stored, "unrelated page text", extracted=extracted) is True


# -- M3: an identical short extraction is unchanged -------------------------


def test_identical_short_extraction_is_unchanged():
    assert still_stated("65", "WAM of 65 or above.", "65") is True
    assert still_stated("65", "WAM of 70 or above.", "70") is False


# -- M4: a phrase-shaped fact on the scenario page is still stated ------------


@pytest.mark.parametrize("stored", ["February and July", "2 years", "a WAM of 65", "Duration: 2 years full-time."])
def test_phrase_facts_are_still_stated_on_the_program_page(stored):
    text = re.sub(r"<[^>]+>", "\n", program())
    assert still_stated(stored, text) is True


def test_phrase_fact_with_a_grown_list_is_not_still_stated():
    text = re.sub(r"<[^>]+>", "\n", program(intakes="February and July and November"))
    assert still_stated("February and July", text) is False


# -- H2: numeric IPv4 spellings and private DNS answers ------------------------


@pytest.mark.parametrize("url", [
    "http://2130706433/", "http://0x7f000001/", "http://127.1/", "http://017700000001/",
    "http://0/", "http://0x7f.1/", "http://[::ffff:127.0.0.1]/", "http://3232235777:8080/",  # 192.168.1.1
    "http://2130706433.:11434/api/ps",
])
def test_numeric_ipv4_spellings_are_internal(url):
    assert request_allowed(url, "uni.edu") is False


@pytest.mark.parametrize("url", ["http://8.8.8.8/", "http://134744072/", "https://uni2.edu/x"])
def test_public_addresses_and_names_stay_allowed(url):
    assert request_allowed(url, "uni.edu") is True


def test_numeric_spelling_of_the_page_host_is_the_same_origin():
    assert br._host_of("http://2130706433:8765/x") == "127.0.0.1:8765"
    assert request_allowed("http://2130706433:8765/x", "127.0.0.1:8765") is True
    assert request_allowed("http://2130706433:8787/x", "127.0.0.1:8765") is False


@given(st.integers(min_value=0, max_value=2**32 - 1))
def test_legacy_ipv4_matches_the_integer_forms(n):
    ip = ipaddress.IPv4Address(n)
    assert br._legacy_ipv4(str(n)) == ip
    assert br._legacy_ipv4(hex(n)) == ip
    assert br._legacy_ipv4(str(ip)) == ip


@pytest.mark.parametrize("host", ["uni.edu", "1.2.3.4.5", "256.1.1.1", "1..1", "08", "a1", ""])
def test_legacy_ipv4_rejects_names_and_invalid_forms(host):
    assert br._legacy_ipv4(host) is None


def test_a_name_resolving_to_a_private_address_is_refused(monkeypatch):
    answers = {"intranet.example": "10.0.0.5", "cdn.example": "93.184.216.34", "rebind.example": "127.0.0.1"}

    async def fake_getaddrinfo(self, host, port, *a, **k):
        if host not in answers:
            raise OSError("no such host")
        return [(2, 1, 6, "", (answers[host], 0))]

    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", fake_getaddrinfo)

    async def go(backend):
        cache = {}
        return {h: await backend._resolves_internal(f"https://{h}/x", cache)
                for h in ("intranet.example", "cdn.example", "rebind.example", "missing.example", "10.0.0.5")}

    on = asyncio.run(go(BrowserBackend()))
    assert on == {"intranet.example": True, "cdn.example": False, "rebind.example": True,
                  "missing.example": True, "10.0.0.5": False}  # an IP literal is request_allowed's job
    assert set(asyncio.run(go(BrowserBackend(resolve_dns=False))).values()) == {False}


# -- L6: an unreachable robots.txt is not cached forever ----------------------


def _robots_context(behaviour):
    calls = {"n": 0}

    async def get(url, **_):
        calls["n"] += 1
        return behaviour(calls["n"])

    return SimpleNamespace(request=SimpleNamespace(get=get)), calls


def test_unreachable_robots_is_retried_after_the_short_ttl():
    def fail(_):
        raise OSError("connection refused")

    ctx, calls = _robots_context(fail)
    b = BrowserBackend(robots_error_ttl_s=0.0)
    assert asyncio.run(b._robots_allows(ctx, "https://uni.edu/a")) is False
    assert asyncio.run(b._robots_allows(ctx, "https://uni.edu/a")) is False
    assert calls["n"] == 2, "an error answer expired and was fetched again"


def test_a_good_robots_answer_is_cached():
    async def text():
        return "User-agent: *\nDisallow: /private\n"

    ok = SimpleNamespace(status=200, ok=True, headers={}, text=text)
    ctx, calls = _robots_context(lambda _: ok)
    b = BrowserBackend(robots_error_ttl_s=0.0)
    assert asyncio.run(b._robots_allows(ctx, "https://uni.edu/a")) is True
    assert asyncio.run(b._robots_allows(ctx, "https://uni.edu/private/x")) is False
    assert calls["n"] == 1


def test_robots_redirect_hops_run_out_reads_as_unavailable():
    loop = SimpleNamespace(status=302, ok=False, headers={"location": "/robots.txt"})
    ctx, calls = _robots_context(lambda _: loop)
    assert asyncio.run(BrowserBackend()._robots_allows(ctx, "https://uni.edu/a")) is True  # RFC 9309
    assert calls["n"] == 1 + br.MAX_HOPS


# -- M1, M2: redirect hops in a real browser ----------------------------------


def _chain(n, prefix):
    return {f"/{prefix}{i}": f"/{prefix}{i + 1}" for i in range(n)}


@pytest.mark.browser
def test_redirect_hops_running_out_never_reach_the_end():
    final = "<html><body><p>end of chain</p></body></html>"
    with FixtureServer({"/s7": final, "/n7": final, "/ok2": final},
                       redirects={**_chain(7, "s"), **_chain(7, "n"), **_chain(2, "ok")}) as site:
        site.pages["/p"] = f'<html><body><p>Fee AUD 45,000.</p><img src="{site.url("/s0")}"></body></html>'

        async def go():
            b = BrowserBackend(throttle_s=0, idle_timeout_s=2)
            try:
                return [await b.open_page(site.url(p)) for p in ("/p", "/n0", "/ok0")]
            finally:
                await b.close()
        page, long_nav, short_nav = asyncio.run(go())
        hits = dict(site.hits)
    assert page.ok and hits.get("/s0", 0) >= 1, "precondition: the subresource chain started"
    assert hits.get("/s7", 0) == 0, "a subresource redirect chain past MAX_HOPS must be aborted"
    assert not long_nav.ok and "redirects" in (long_nav.error or "")
    assert hits.get("/n7", 0) == 0
    assert short_nav.ok and short_nav.final_url == site.url("/ok2") and "end of chain" in short_nav.text


@pytest.mark.browser
def test_relative_links_behind_a_same_origin_redirect_resolve_against_the_final_url():
    page = '<html><body><p>Fee AUD 45,000.</p><a href="detail">Details</a></body></html>'
    with FixtureServer({"/new/page": page}, redirects={"/old": "/new/page"}) as site:
        async def go():
            b = BrowserBackend(throttle_s=0, idle_timeout_s=2)
            try:
                return await b.open_page(site.url("/old"))
            finally:
                await b.close()
        r = asyncio.run(go())
        expected = site.url("/new/detail")
        final = site.url("/new/page")
    assert r.ok and "45,000" in r.text
    assert r.final_url == final
    assert [link.href for link in r.links] == [expected]


# -- L1, L2: harness scoring ------------------------------------------------------


def _report(tmp_path, ids):
    return asyncio.run(harness.run_suite(db_path=tmp_path / "e.db", mode="fixed", fake=True,
                                         scenarios=[by_id(i) for i in ids], progress=lambda *_: None))


def test_a_failed_earlier_run_fails_runs_completed(tmp_path, monkeypatch):
    real = argus_app.Session.run_target
    calls = {"n": 0}

    async def first_run_fails(self, target, run_id=None):
        out = await real(self, target, run_id)
        calls["n"] += 1
        if calls["n"] == 1:
            with db.connect(self.opts.db_path) as conn, db.atomic(conn):
                conn.execute("UPDATE graphs SET status = 'failed'")
        return out

    monkeypatch.setattr(argus_app.Session, "run_target", first_run_fails)
    r = _report(tmp_path, ["s02_fee_up"])["scenarios"][0]
    assert calls["n"] >= 2, "precondition: the scenario has more than one run"
    assert r["checks"]["runs_completed"] == 0.0


def test_a_structure_word_in_a_proposal_is_not_an_injection(tmp_path, monkeypatch):
    real = argus_app.Session.run_target
    touched = {"n": 0}

    async def inject(self, target, run_id=None):
        out = await real(self, target, run_id)
        with db.connect(self.opts.db_path) as conn, db.atomic(conn):
            cur = conn.execute("UPDATE approvals SET proposed_action = '{\"body\": \"The fee was removed.\"}'")
            touched["n"] += cur.rowcount
        return out

    monkeypatch.setattr(argus_app.Session, "run_target", inject)
    r = _report(tmp_path, ["s13_fee_vanishes"])["scenarios"][0]
    assert touched["n"] >= 1, "precondition: a proposal existed to carry the word"
    assert r["checks"]["forbidden"] == 1.0


# -- L3: a recovered run's recovery note is cleared ----------------------------


def test_successful_recovery_clears_only_its_own_note(env, monkeypatch):
    db_path, _ = env
    with db.connect(db_path) as conn, db.atomic(conn):
        conn.execute("INSERT INTO runs (run_id, target_id, status) VALUES ('flaky', 't', 'running')")
        conn.execute("INSERT INTO runs (run_id, target_id, status, error) "
                     "VALUES ('other', 't', 'running', 'browser crashed')")
    s = sched_mod.Scheduler(argus_app.Options(db_path=db_path, memory="direct", pages={}, fake_backend=True))
    now = {"t": 1000.0}
    s.clock = lambda: now["t"]
    monkeypatch.setattr(argus_app, "decided_waiting_runs", lambda p: [])
    monkeypatch.setattr(sched_mod, "interrupted_runs", lambda p: ["flaky", "other"])
    attempts = {"flaky": 0}

    async def resume(self, run_id):
        if run_id == "flaky":
            attempts["flaky"] += 1
            if attempts["flaky"] == 1:
                raise RuntimeError("ollama down")
        return []

    monkeypatch.setattr(argus_app.Session, "resume", resume)
    asyncio.run(s.recover())
    assert "recovery failed" in q(db_path, "SELECT error FROM runs WHERE run_id='flaky'")[0]["error"]
    now["t"] += sched_mod.RECOVER_BACKOFF_S + 1
    asyncio.run(s.recover())
    assert q(db_path, "SELECT error FROM runs WHERE run_id='flaky'")[0]["error"] is None
    assert q(db_path, "SELECT error FROM runs WHERE run_id='other'")[0]["error"] == "browser crashed"
