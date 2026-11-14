"""Browser MCP server: real Chromium against a local fixture server.

Layers (argus-mcp-testing skill): backend in-process, then a real stdio
subprocess through the orchestrator's McpBrowser client.
"""

from __future__ import annotations

import asyncio
import sys
import time

import pytest

from argus.eval.fixture_server import FixtureServer
from argus.graph.mcp_clients import McpBrowser
from argus.servers.browser import BrowserBackend

pytestmark = pytest.mark.browser

HTML = """<html><head><title>Master of AI</title></head><body>
<h1>Master of AI</h1><p>The annual tuition fee for 2027 is AUD 45,000.</p>
<a href="/uni/fees">Fees and scholarships</a> <a href="https://elsewhere.example/x">Partner</a>
<a href="javascript:void(0)">Menu</a></body></html>"""
JS = """<html><body><div id="x">loading</div>
<script>setTimeout(() => { document.getElementById('x').innerText = 'Intakes: February and July.' }, 200)</script>
</body></html>"""


def run(coro):
    return asyncio.run(coro)


async def _with_backend(fn, **kw):
    b = BrowserBackend(throttle_s=kw.pop("throttle_s", 0.0), **kw)
    try:
        return await fn(b)
    finally:
        await b.close()


def test_page_text_title_and_absolute_links():
    with FixtureServer({"/uni/ai": HTML}) as site:
        r = run(_with_backend(lambda b: b.open_page(site.url("/uni/ai"))))
    assert r.ok and r.status == 200 and not r.partial and r.confidence == 1.0
    assert "AUD 45,000" in r.text and r.title == "Master of AI"
    hrefs = [l.href for l in r.links]
    assert site.url("/uni/fees") in hrefs and "https://elsewhere.example/x" in hrefs
    assert not any(h.startswith("javascript:") for h in hrefs)


def test_script_rendered_text_is_captured_by_waiting_for_the_network_to_settle():
    with FixtureServer({"/js": JS}) as site:
        r = run(_with_backend(lambda b: b.open_page(site.url("/js")), idle_timeout_s=5))
    assert "February and July" in r.text


@pytest.mark.parametrize("path,page", [("/gone", None), ("/err", (500, "<html><body>oops</body></html>"))])
def test_http_errors_degrade_instead_of_raising(path, page):
    pages = {path: page} if page else {}
    with FixtureServer(pages) as site:
        r = run(_with_backend(lambda b: b.open_page(site.url(path))))
    assert not r.ok and r.partial and r.confidence == 0.0 and r.status in (404, 500)


def test_unreachable_host_and_bad_schemes_degrade():
    async def go(b):
        return [await b.open_page("http://127.0.0.1:9/never"), await b.open_page("file:///C:/Windows/win.ini"),
                await b.open_page("not a url")]
    results = run(_with_backend(go, timeout_s=5))
    assert all(not r.ok and r.confidence == 0.0 for r in results)


def test_robots_disallow_is_honoured():
    with FixtureServer({"/private/x": HTML}, robots="User-agent: *\nDisallow: /private/\n") as site:
        r = run(_with_backend(lambda b: b.open_page(site.url("/private/x"))))
    assert not r.ok and "robots" in r.error and site.hits["/private/x"] == 0


def test_contexts_are_always_closed():
    async def go(b):
        with FixtureServer({"/a": HTML}) as site:
            await b.open_page(site.url("/a"))
            await b.open_page(site.url("/missing"))
        await b.open_page("http://127.0.0.1:9/x")
        return b.open_contexts
    assert run(_with_backend(go, timeout_s=5)) == 0


def test_same_host_requests_are_throttled():
    with FixtureServer({"/a": HTML, "/b": HTML}) as site:
        async def go(b):
            await b.open_page(site.url("/a"))
            await b.open_page(site.url("/b"))
        run(_with_backend(go, throttle_s=1.0))
        gap = site.hit_times["/b"][0] - site.hit_times["/a"][0]
    assert gap >= 0.95


def test_slow_page_returns_partial_text_rather_than_hanging():
    slow_js = """<html><body><p>Fees: AUD 45,000.</p>
<script>setInterval(() => fetch('/poll'), 100)</script></body></html>"""
    with FixtureServer({"/slow": slow_js, "/poll": "ok"}) as site:
        start = time.monotonic()
        r = run(_with_backend(lambda b: b.open_page(site.url("/slow")), idle_timeout_s=1.5))
        elapsed = time.monotonic() - start
    assert r.ok and r.partial and r.confidence == pytest.approx(0.7) and "45,000" in r.text
    assert elapsed < 10


def test_over_stdio_through_the_orchestrator_client():
    with FixtureServer({"/uni/ai": HTML}) as site:
        async def go():
            client = McpBrowser(["--throttle-s", "0"])
            await client.open()
            try:
                return await client.open_page(site.url("/uni/ai")), await client.open_page(site.url("/nope"))
            finally:
                await client.close()
        good, bad = run(go())
    assert good["ok"] and "AUD 45,000" in good["text"]
    assert bad["ok"] is False and bad["status"] == 404
