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
  - Network boundary (Phase F review M2): every request the page makes is
    filtered. Loopback, private, link-local and `localhost` addresses are
    blocked unless they are the requested page's own host (the local fixture
    site is), and a redirect to a different host is refused, so a watched
    page cannot point the browser at the dashboard, Ollama or a LAN device.
  - Bounded (review M3): reading the page after load is time-boxed, so a
    page that busy-loops degrades instead of hanging the server.
The orchestrator decides WHICH urls are allowed (same site as the target);
this server only decides HOW to fetch politely.
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
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


def _host_of(url: str) -> str:
    """Origin key: host (lowercase, punycode, trailing dot and www. folded),
    plus the port ONLY when it is explicit and not the scheme's default.

    So http://uni.edu and https://uni.edu are the same site (an http->https
    upgrade is allowed, matching handlers.same_site; delta review M-G), while
    127.0.0.1:8765 (a fixture site) and 127.0.0.1:8787 (the dashboard) are
    different origins. IPv6 hosts are bracketed so the port split is
    unambiguous."""
    try:
        u = urlparse(url)
        h = (u.hostname or "").lower().rstrip(".")
        port = u.port
    except ValueError:
        return ""
    if not h:
        return ""
    try:
        h = h.encode("idna").decode("ascii")  # page.url is punycode; compare like with like
    except UnicodeError:
        pass
    h = h[4:] if h.startswith("www.") else h
    if ":" in h:
        h = f"[{h}]"
    if port is None or port == {"http": 80, "https": 443}.get(u.scheme):
        return h
    return f"{h}:{port}"


def _internal(key: str) -> bool:
    """True for loopback, private, link-local, CGNAT and every other
    non-global address, and for localhost names (trailing dots folded)."""
    host = key
    if host.startswith("["):
        host = host[1:host.index("]")]
    elif host.count(":") == 1:
        host = host.rsplit(":", 1)[0]
    host = host.rstrip(".")
    if host in ("localhost", "localhost.localdomain") or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return not ip.is_global


def request_allowed(request_url: str, page_host: str) -> bool:
    """A request made while rendering a page may go to the page's own host,
    or to any PUBLIC host (scripts, styles). Never to another internal one."""
    host = _host_of(request_url)
    if not host:
        return False
    if host == page_host:
        return True
    return not _internal(host)


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
                 idle_timeout_s: float = 8.0, respect_robots: bool = True, read_timeout_s: float = 15.0):
        self.throttle_s = throttle_s
        self.read_timeout_s = read_timeout_s
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
            if self._browser is not None and not self._browser.is_connected():
                log("chromium disconnected; relaunching")  # review M4
                await self._drop_browser()
            if self._browser is None:
                from playwright.async_api import async_playwright

                self._pw = await async_playwright().start()
                self._browser = await self._pw.chromium.launch(headless=True)
                log("chromium launched")
        return self._browser

    async def _drop_browser(self) -> None:
        """Forget the current Chromium AND stop its Playwright driver, so a
        relaunch does not leak the old processes (delta review, low)."""
        browser, pw, self._browser, self._pw = self._browser, self._pw, None, None
        for closer in ((browser.close if browser is not None else None), (pw.stop if pw is not None else None)):
            if closer is None:
                continue
            try:
                await asyncio.wait_for(closer(), timeout=10)
            except Exception as exc:  # noqa: BLE001 - it is already gone or wedged
                log(f"cleanup of old browser failed: {exc!r}")

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
                # No automatic redirects (review H-A): follow at most five
                # same-origin hops ourselves; a cross-origin robots redirect
                # is not followed and reads as "no robots.txt".
                target = origin + "/robots.txt"
                resp = await context.request.get(target, timeout=10_000, max_redirects=0)
                for _ in range(5):
                    if resp.status not in (301, 302, 303, 307, 308):
                        break
                    nxt = urljoin(target, resp.headers.get("location", ""))
                    if _host_of(nxt) != _host_of(origin):
                        break
                    target = nxt
                    resp = await context.request.get(target, timeout=10_000, max_redirects=0)
                if resp.ok:
                    parser = urllib.robotparser.RobotFileParser()
                    parser.parse((await resp.text()).splitlines())
                elif resp.status >= 500:
                    # RFC 9309: a server error on robots.txt means disallow.
                    parser = urllib.robotparser.RobotFileParser()
                    parser.parse(["User-agent: *", "Disallow: /"])
            except Exception as exc:  # noqa: BLE001 - RFC 9309: unreachable means disallow
                log(f"robots.txt unreachable for {origin}: {exc!r}; treating as disallow")
                parser = urllib.robotparser.RobotFileParser()
                parser.parse(["User-agent: *", "Disallow: /"])
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
        try:
            context = await browser.new_context(user_agent=USER_AGENT, accept_downloads=False,
                                                java_script_enabled=True)
        except Exception as exc:  # noqa: BLE001 - Chromium died: relaunch next time, degrade now
            await self._drop_browser()
            return PageResult(url=url, ok=False, partial=True, confidence=0.0,
                              error=f"browser context failed: {exc!r}", elapsed_s=time.monotonic() - start)
        self.open_contexts += 1
        page_host = _host_of(url)
        blocked: List[str] = []

        async def guard(route):
            """Every request, every redirect hop (delta review H-A).

            Playwright calls a route handler only for the FIRST url of a
            redirect chain, so letting the browser follow redirects would let
            a public URL bounce a request into the LAN. Instead the request is
            fetched with redirects off and each Location is checked before
            the next hop. A top-level navigation may only move within its own
            origin."""
            req = route.request
            url = req.url
            if not request_allowed(url, page_host):
                blocked.append(url)
                await route.abort("blockedbyclient")
                return
            try:
                resp = await route.fetch(max_redirects=0)
                for _ in range(5):
                    if resp.status not in (301, 302, 303, 307, 308):
                        break
                    nxt = urljoin(url, resp.headers.get("location", ""))
                    if not request_allowed(nxt, page_host) or (
                            req.is_navigation_request() and _host_of(nxt) != page_host):
                        blocked.append(nxt)
                        await route.abort("blockedbyclient")
                        return
                    url = nxt
                    resp = await route.fetch(url=url, max_redirects=0)
                await route.fulfill(response=resp)
            except Exception as exc:  # noqa: BLE001 - one failed subresource is not fatal
                blocked.append(f"{url} ({type(exc).__name__})")
                try:
                    await route.abort("failed")
                except Exception:  # noqa: BLE001 - already handled
                    pass

        try:
            await context.route("**/*", guard)
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
            if _host_of(page.url) != page_host:
                return PageResult(url=url, final_url=page.url, ok=False, partial=True, confidence=0.0,
                                  error=f"redirected to another origin ({_host_of(page.url)}); refused",
                                  elapsed_s=time.monotonic() - start)
            settled = True
            try:
                await page.wait_for_load_state("networkidle", timeout=self.idle_timeout_ms)
            except Exception:  # noqa: BLE001 - the network never went quiet
                settled = False
            status = response.status if response is not None else None
            page.set_default_timeout(int(self.read_timeout_s * 1000))

            async def read():
                title = await page.title()
                text = (await page.inner_text("body"))[:MAX_TEXT_CHARS] if await page.query_selector("body") else ""
                raw = await page.eval_on_selector_all(
                    "a[href]", "els => els.map(e => [e.innerText || e.textContent || '', e.getAttribute('href')])")
                return title, text, raw

            try:
                title, text, raw_links = await asyncio.wait_for(read(), timeout=self.read_timeout_s)
            except asyncio.TimeoutError:
                return PageResult(url=url, final_url=page.url, status=status, ok=False, partial=True,
                                  confidence=0.0, error=f"page did not respond within {self.read_timeout_s}s "
                                  "after loading (script busy?)", elapsed_s=time.monotonic() - start)
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
                await asyncio.wait_for(context.close(), timeout=10)
            except Exception as exc:  # noqa: BLE001 - a wedged renderer: drop this browser
                log(f"context close failed ({exc!r}); browser will be relaunched")
                self._browser = None
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
