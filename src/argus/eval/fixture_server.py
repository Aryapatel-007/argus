"""A tiny local web server for fixture pages (tests and the eval harness).

Binds 127.0.0.1 on a free port. Pages are a mutable dict, so a scenario can
change a page between two runs ("the fee went up") and Argus sees a real
HTTP change through a real browser. Nothing here touches the internet.

    with FixtureServer({"/uni/ai": "<html>...</html>"}) as site:
        site.url("/uni/ai")              # http://127.0.0.1:PORT/uni/ai
        site.pages["/uni/ai"] = "<html>new version</html>"
        site.hits["/uni/ai"]             # request count
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional, Tuple, Union

Page = Union[str, Tuple[int, str]]


class FixtureServer:
    def __init__(self, pages: Optional[Dict[str, Page]] = None, robots: str = "User-agent: *\nAllow: /\n",
                 slow_paths: Optional[Dict[str, float]] = None, redirects: Optional[Dict[str, str]] = None):
        self.pages: Dict[str, Page] = dict(pages or {})
        #: path -> absolute Location, answered with a real HTTP 302
        self.redirects: Dict[str, str] = dict(redirects or {})
        self.robots = robots
        self.slow_paths = dict(slow_paths or {})
        self.hits: Dict[str, int] = defaultdict(int)
        self.hit_times: Dict[str, list] = defaultdict(list)
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    def __enter__(self) -> "FixtureServer":
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output (and stdout) clean
                pass

            def do_GET(self):  # noqa: N802
                path = self.path.split("?", 1)[0]
                owner.hits[path] += 1
                owner.hit_times[path].append(time.monotonic())
                if path == "/robots.txt":
                    return self._send(200, owner.robots, "text/plain")
                if path in owner.redirects:
                    self.send_response(302)
                    self.send_header("Location", owner.redirects[path])
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return None
                if path in owner.slow_paths:
                    time.sleep(owner.slow_paths[path])  # simulates a slow origin
                page = owner.pages.get(path)
                if page is None:
                    return self._send(404, "<html><body>not found</body></html>")
                status, body = page if isinstance(page, tuple) else (200, page)
                return self._send(status, body)

            def _send(self, status: int, body: str, ctype: str = "text/html; charset=utf-8"):
                data = body.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
