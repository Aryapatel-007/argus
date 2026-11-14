"""What node handlers talk to: model backend, memory, browser, actions.

Each is a small async interface with two kinds of implementation:
  - in-process (`DirectMemory`, `StaticBrowser`) for unit tests and the eval
    harness's deterministic mode
  - real (`argus.graph.mcp_clients`): the memory and browser MCP servers over
    stdio, spawned per run (Phase F)

The engine never imports a backend directly; it only sees `Services`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from argus.graph.llm import ModelBackend


class MemoryAPI(Protocol):
    async def list_current_facts(self, target_id: str) -> List[Dict[str, Any]]: ...
    async def recall(self, query: str, target_id: str, limit: int = 5) -> List[Dict[str, Any]]: ...
    async def remember(self, **fact: Any) -> Dict[str, Any]: ...


class BrowserAPI(Protocol):
    async def open_page(self, url: str) -> Dict[str, Any]: ...


class DirectMemory:
    """MemoryServer backend called in-process. Tests and offline eval only:
    production goes through the MCP server (mcp_clients.McpMemory)."""

    def __init__(self, db_path: Path, run_id: str):
        from argus.servers.memory import MemoryServer

        self.backend = MemoryServer(Path(db_path), run_id)

    async def list_current_facts(self, target_id: str) -> List[Dict[str, Any]]:
        return [f.model_dump() for f in await self.backend.list_current_facts(target_id)]

    async def recall(self, query: str, target_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        return [f.model_dump() for f in await self.backend.recall_facts(query, target_id=target_id, limit=limit)]

    async def remember(self, **fact: Any) -> Dict[str, Any]:
        return (await self.backend.remember_fact(**fact)).model_dump()


class StaticBrowser:
    """Pages from a dict. Missing URL -> the same degraded shape the real
    browser server returns for an unreachable page."""

    def __init__(self, pages: Dict[str, Dict[str, Any]]):
        self.pages = pages
        self.opened: List[str] = []

    async def open_page(self, url: str) -> Dict[str, Any]:
        self.opened.append(url)
        page = self.pages.get(url)
        if page is None:
            return {"url": url, "final_url": url, "status": 404, "title": "", "text": "",
                    "links": [], "ok": False, "partial": True, "confidence": 0.0,
                    "error": "not found"}
        return {"url": url, "final_url": url, "status": 200, "ok": True, "partial": False,
                "confidence": 1.0, "error": None, "links": [], **page}


class ActionExecutor:
    """Performs approved external actions. Exactly once, by idempotency key.

    `outbox_message` writes data/outbox/<key>.json. The file name IS the
    idempotency key, so a resumed execute after a crash finds the file and
    does not send twice. A real outbound channel (email, webhook) would need a
    credential: that is a BLOCKED: needs Arya decision, not built here.
    """

    def __init__(self, outbox: Path):
        self.outbox = Path(outbox)

    def already_done(self, key: str) -> bool:
        return (self.outbox / f"{key}.json").exists()

    def execute(self, kind: str, payload: Dict[str, Any], key: str) -> Dict[str, Any]:
        if kind != "outbox_message":
            raise ValueError(f"no executor for action kind {kind!r}")
        self.outbox.mkdir(parents=True, exist_ok=True)
        path = self.outbox / f"{key}.json"
        if path.exists():
            return {"written": str(path), "duplicate": True}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"idempotency_key": key, **payload}, indent=2, sort_keys=True),
                       encoding="utf-8")
        os.replace(tmp, path)  # atomic: a crash leaves either no file or the whole file
        return {"written": str(path), "duplicate": False}


@dataclass
class Services:
    backend: ModelBackend
    memory: MemoryAPI
    browser: BrowserAPI
    actions: ActionExecutor
