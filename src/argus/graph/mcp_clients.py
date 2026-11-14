"""The orchestrator's MCP clients (Phase F 1 and 3).

The memory server and the browser server run as SEPARATE processes, spoken
to over stdio with the MCP v2 SDK. The engine never imports their backends
in production; it only sees MemoryAPI / BrowserAPI (graph/services.py).

Lifecycle (ARGUS_BUILD 7.1):
  - spawned per run, AFTER the runs row is committed (the memory server
    refuses a --run-id that does not exist)
  - a resumed run gets a FRESH spawn, never a reused process
  - `close()` ends the session and the child; on Windows the MCP stdio
    client puts the child in a Job Object, and tests verify by process
    listing that no server survives its parent (argus-mcp-testing skill)

Errors: a tool that fails comes back as is_error with text. Retryable
failures (timeouts, a locked database) and permanent ones are told apart by
message here, because the server does not type them (audit M7, backlog).
"""

from __future__ import annotations

import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client

#: Seconds to wait for one tool call. Memory tools embed (~1s); the browser
#: renders pages. Both stay well under the 120s model cap.
READ_TIMEOUT_S = 90.0


class ToolCallError(RuntimeError):
    def __init__(self, tool: str, text: str):
        self.tool = tool
        self.retryable = any(s in text.lower() for s in ("timeout", "locked", "busy", "unreachable"))
        super().__init__(f"{tool}: {text}")


def _server_env() -> Dict[str, str]:
    # Inherit the environment: PYTHONPATH (tests, mutation runs), SystemRoot
    # (Windows needs it to start Python), ARGUS_* switches.
    return dict(os.environ)


class _StdioTool:
    def __init__(self, module: str, args: List[str]):
        self.module = module
        self.args = args
        self._stack: Optional[AsyncExitStack] = None
        self.client: Optional[Client] = None

    async def open(self) -> None:
        if self.client is not None:
            return
        params = StdioServerParameters(command=sys.executable, args=["-m", self.module, *self.args],
                                       env=_server_env())
        stack = AsyncExitStack()
        try:
            self.client = await stack.enter_async_context(
                Client(stdio_client(params), read_timeout_seconds=READ_TIMEOUT_S))
        except BaseException:
            await stack.aclose()
            raise
        self._stack = stack

    async def close(self) -> None:
        stack, self._stack, self.client = self._stack, None, None
        if stack is not None:
            await stack.aclose()

    async def call(self, tool: str, args: Dict[str, Any]) -> Any:
        if self.client is None:
            raise RuntimeError(f"{self.module} is not open")
        result = await self.client.call_tool(tool, {k: v for k, v in args.items() if v is not None})
        if result.is_error:
            text = " ".join(getattr(c, "text", "") for c in result.content or [])
            raise ToolCallError(tool, text or "tool error")
        sc = result.structured_content
        if isinstance(sc, dict) and set(sc) == {"result"}:
            return sc["result"]
        return sc


class McpMemory(_StdioTool):
    """MemoryAPI over the argus-memory MCP server."""

    def __init__(self, db_path: Path, run_id: str):
        super().__init__("argus.servers.memory", ["--db-path", str(db_path), "--run-id", run_id])

    async def list_current_facts(self, target_id: str) -> List[Dict[str, Any]]:
        return await self.call("list_current_facts", {"target_id": target_id})

    async def recall(self, query: str, target_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        return await self.call("recall_facts", {"query": query, "target_id": target_id, "limit": limit})

    async def remember(self, **fact: Any) -> Dict[str, Any]:
        return await self.call("remember_fact", fact)


class McpBrowser(_StdioTool):
    """BrowserAPI over the argus-browser MCP server."""

    def __init__(self, extra_args: Optional[List[str]] = None):
        super().__init__("argus.servers.browser", list(extra_args or []))

    async def open_page(self, url: str) -> Dict[str, Any]:
        # The owner must open() first, in the task that will close() it
        # (anyio cancel scopes are task-bound). No lazy open here.
        return await self.call("open_page", {"url": url})


class RunMemory:
    """Binds a MemoryAPI to the run once its id exists, and owns its lifetime.

    `mode='mcp'` (production): a fresh McpMemory per run/resume.
    `mode='direct'`: in-process backend (tests, offline eval).
    """

    def __init__(self, db_path: Path, mode: str = "mcp", hang_after: Optional[int] = None):
        self.db_path = Path(db_path)
        self.mode = mode
        self.inner: Any = None
        self.hang_after = hang_after
        self.remembered = 0

    async def open(self, run_id: str) -> None:
        await self.close()
        if self.mode == "mcp":
            self.inner = McpMemory(self.db_path, run_id)
            await self.inner.open()
        else:
            from argus.graph.services import DirectMemory

            self.inner = DirectMemory(self.db_path, run_id)

    async def close(self) -> None:
        inner, self.inner = self.inner, None
        if isinstance(inner, McpMemory):
            await inner.close()

    async def list_current_facts(self, target_id):
        return await self.inner.list_current_facts(target_id)

    async def recall(self, query, target_id, limit=5):
        return await self.inner.recall(query, target_id, limit)

    async def remember(self, **kw):
        result = await self.inner.remember(**kw)
        self.remembered += 1
        if self.hang_after is not None and self.remembered >= self.hang_after:
            import asyncio

            await asyncio.sleep(3600)  # crash tests: kill lands after the commit
        return result
