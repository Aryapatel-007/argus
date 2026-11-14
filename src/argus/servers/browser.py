"""Argus browser MCP server: raw Playwright, page text out (D-002).

Run: python -m argus.servers.browser [--throttle-s 2.0] [--timeout-s 30]

One tool, `open_page(url)`. It renders the page in headless Chromium and
returns its visible text, title and links. It NEVER raises to the caller for
a page problem: an unreachable, slow, blocked or broken page comes back as a
result with ok=false, partial=true and a confidence score, so one bad page
degrades a run instead of killing it (ARGUS_BUILD 7.3).

Rules this file keeps:
  - STDOUT BELONGS TO MCP. Diagnostics go to stderr only.
  - Real wait conditions, never fixed sleeps: `domcontentloaded`, then a
    bounded wait for `networkidle`. If the network never settles the text is
    still returned, marked partial.
  - Every call gets its own browser context, closed in `finally`. One
    Chromium per server process, closed by the lifespan on shutdown.
  - Per-domain self-throttle: at most one request per `throttle_s` per host
    (a politeness rate limit, not a readiness wait).
  - robots.txt is honoured for our user agent.
  - http(s) only. No downloads, no file://, no data:.
The orchestrator decides WHICH urls are allowed (same site as the target);
this server only decides HOW to fetch politely.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import urllib.robotparser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

if __package__ in (None, ""):  # direct-script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp.server import MCPServer  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

SERVER_NAME = "argus-browser"
USER_AGENT = "ArgusMonitor/0.3 (local personal monitoring; +robots.txt respected)"
MAX_TEXT_CHARS = 200_000
MAX_LINKS = 200


def log(message: str) -> None:
    print(f"[{SERVER_NAME}] {message}", file=sys.stderr, flush=True)


class Link(BaseModel):
    text: str
    href: str


class PageResult(BaseModel):
    url: str
    final_url: Optional[str] = None
    status: Optional[int] = None
    title: str = ""
    text: str = ""
    links: List[Link] = Field(default_factory=list)
    ok: bool
    partial: bool = Field(description="True if the page may be incomplete.")
    confidence: float = Field(ge=0.0, le=1.0, description="How complete the capture is.")
    error: Optional[str] = None
    elapsed_s: float = 0.0


class BrowserBackend:
    def __init__(self, *, throttle_s: float = 2.0, timeout_s: float = 30.0,
                 idle_timeout_s: float = 8.0, respect_robots: bool = True):
        self.throttle_s = throttle_s
        self.timeout_ms = int(timeout_s * 1000)
        self.idle_timeout_ms = int(idle_timeout_s * 1000)
        self.respect_robots = respect_robots
        self._pw = None
        self._browser = None
        self._launch_lock = asyncio.Lock()
        self._host_locks: Dict[str, asyncio.Lock] = {}
        self._host_last: Dict[str, float] = {}
        self._robots: Dict[str, Optional[urllib.robotparser.RobotFileParser]] = {}
        self.open_contexts = 0  # tests assert this returns to 0

    async def _ensure_browser(self):
        async with self._launch_lock:
            if self._browser is None:
                from playwright.async_api import async_playwright

                self._pw = await async_playwright().start()
                self._browser = await self._pw.chromium.launch(headless=True)
                log("chromium launched")
        return self._browser

    async def close(self) -> None:
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._pw is not None:
            await self._pw.stop()
            self._pw = None

    async def _throttle(self, host: str) -> None:
        lock = self._host_locks.setdefault(host, asyncio.Lock())
        async with lock:
            wait = self.throttle_s - (time.monotonic() - self._host_last.get(host, -1e9))
            if wait > 0:
                await asyncio.sleep(wait)  # rate limit per host, not a readiness wait
            self._host_last[host] = time.monotonic()

    async def _robots_allows(self, context, url: str) -> bool:
        if not self.respect_robots:
            return True
        parts = urlparse(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            parser: Optional[urllib.robotparser.RobotFileParser] = None
            try:
                resp = await context.request.get(origin + "/robots.txt", timeout=10_000)
                if resp.ok:
                    parser = urllib.robotparser.RobotFileParser()
                    parser.parse((await resp.text()).splitlines())
            except Exception as exc:  # noqa: BLE001 - no robots.txt reachable: allowed
                log(f"robots.txt unavailable for {origin}: {exc!r}")
            self._robots[origin] = parser
        parser = self._robots[origin]
        return parser is None or parser.can_fetch(USER_AGENT, url)

    async def open_page(self, url: str) -> PageResult:
        start = time.monotonic()
        parts = urlparse(url or "")
        if parts.scheme not in ("http", "https") or not parts.hostname:
            return PageResult(url=url or "", ok=False, partial=True, confidence=0.0,
                              error="only absolute http(s) urls are allowed")
        try:
            browser = await self._ensure_browser()
        except Exception as exc:  # noqa: BLE001
            return PageResult(url=url, ok=False, partial=True, confidence=0.0,
                              error=f"browser unavailable: {exc!r}")
        await self._throttle(parts.hostname)
        context = await browser.new_context(user_agent=USER_AGENT, accept_downloads=False,
                                            java_script_enabled=True)
        self.open_contexts += 1
        try:
            if not await self._robots_allows(context, url):
                return PageResult(url=url, ok=False, partial=True, confidence=0.0,
                                  error="disallowed by robots.txt", elapsed_s=time.monotonic() - start)
            page = await context.new_page()
            try:
                response = await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            except Exception as exc:  # noqa: BLE001 - unreachable, DNS, timeout
                return PageResult(url=url, ok=False, partial=True, confidence=0.0,
                                  error=f"navigation failed: {type(exc).__name__}: {str(exc)[:300]}",
                                  elapsed_s=time.monotonic() - start)
            settled = True
            try:
                await page.wait_for_load_state("networkidle", timeout=self.idle_timeout_ms)
            except Exception:  # noqa: BLE001 - the network never went quiet
                settled = False
            status = response.status if response is not None else None
            title = await page.title()
            text = (await page.inner_text("body"))[:MAX_TEXT_CHARS] if await page.query_selector("body") else ""
            raw_links = await page.eval_on_selector_all(
                "a[href]", "els => els.map(e => [e.innerText || e.textContent || '', e.getAttribute('href')])")
            base = page.url
            links = []
            for label, href in raw_links[:MAX_LINKS * 2]:
                if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
                    continue
                try:
                    absolute = urljoin(base, href)
                except ValueError:
                    continue
                links.append(Link(text=" ".join((label or "").split())[:200], href=absolute))
                if len(links) >= MAX_LINKS:
                    break
            ok = status is not None and 200 <= status < 400 and bool(text.strip())
            confidence = (1.0 if settled else 0.7) if ok else 0.0
            return PageResult(url=url, final_url=base, status=status, title=title, text=text,
                              links=links, ok=ok, partial=not settled or not ok, confidence=confidence,
                              error=None if ok else f"status {status}, {len(text.strip())} chars of text",
                              elapsed_s=time.monotonic() - start)
        except Exception as exc:  # noqa: BLE001 - never kill the caller's run
            return PageResult(url=url, ok=False, partial=True, confidence=0.0,
                              error=f"{type(exc).__name__}: {str(exc)[:300]}", elapsed_s=time.monotonic() - start)
        finally:
            try:
                await context.close()
            finally:
                self.open_contexts -= 1


def build_server(backend: BrowserBackend) -> MCPServer:
    @asynccontextmanager
    async def lifespan(_server):
        try:
            yield None
        finally:
            await backend.close()
            log("chromium closed")

    mcp = MCPServer(SERVER_NAME, lifespan=lifespan)

    @mcp.tool(name="open_page", description=(
        "Open one web page in a headless browser and return its visible text, title and links. "
        "Never fails for a page problem: check `ok`, `partial` and `confidence`."))
    async def open_page(url: str) -> PageResult:
        return await backend.open_page(url)

    return mcp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="argus.servers.browser")
    ap.add_argument("--throttle-s", type=float, default=2.0)
    ap.add_argument("--timeout-s", type=float, default=30.0)
    ap.add_argument("--no-robots", action="store_true", help="tests against local fixtures only")
    args = ap.parse_args(argv)
    backend = BrowserBackend(throttle_s=args.throttle_s, timeout_s=args.timeout_s,
                             respect_robots=not args.no_robots)
    log("serving")
    build_server(backend).run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
