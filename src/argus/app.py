"""High-level operations shared by the CLI, the scheduler and the dashboard.

    run_target(target)     one watch run: one graph, its own memory server
    watch_all()            every target in config/targets.yaml, one after another
    resume(run_id)         drive an unfinished or approved-and-waiting run on
    resume_decided()       every waiting run whose approvals a human decided
    decide(approval_id)    approve or reject (the human's half of the gate)

Each run gets a fresh memory MCP server (spawned after its runs row exists)
and the browser MCP server; both are opened and closed in the calling task.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from argus import config
from argus.graph import fakes
from argus.graph.engine import Engine, RunOutcome
from argus.graph.llm import OllamaBackend
from argus.graph.mcp_clients import McpBrowser, RunMemory
from argus.graph.services import ActionExecutor, Services, StaticBrowser
from argus.storage import db, lease
from argus.storage.migrate import apply_migrations


@dataclass
class Options:
    db_path: Path = db.DB_PATH
    mode: str = "fixed"
    models: Optional[List[str]] = None
    memory: str = "mcp"            # mcp | direct
    pages: Optional[Dict[str, Any]] = None   # StaticBrowser instead of Chromium
    fake_backend: bool = False
    fake_embed: bool = False
    outbox: Optional[Path] = None
    browser_args: List[str] = field(default_factory=list)
    #: crash-test hooks (runner flags), never set in production
    backend: Any = None
    actions: Any = None
    browser: Any = None             # any BrowserAPI (eval harness)
    hang_remember_after: Optional[int] = None


def _services(opts: Options):
    if opts.backend is not None:
        backend = opts.backend
    elif opts.fake_backend:
        backend = fakes.FakeBackend()
    else:
        backend = OllamaBackend()
    if opts.fake_embed:
        fakes.install_fake_embeddings()
        os.environ["ARGUS_FAKE_EMBED"] = "1"  # inherited by the memory server
    if opts.browser is not None:
        browser = opts.browser
    else:
        browser = StaticBrowser(opts.pages) if opts.pages is not None else McpBrowser(opts.browser_args)
    memory = RunMemory(Path(opts.db_path), mode=opts.memory, hang_after=opts.hang_remember_after)
    outbox = Path(opts.outbox) if opts.outbox else Path(opts.db_path).parent / "outbox"
    actions = opts.actions or ActionExecutor(outbox)
    return Services(backend=backend, memory=memory, browser=browser, actions=actions), memory


class Session:
    """Owns the per-run servers. Use as `async with Session(opts) as s:`."""

    def __init__(self, opts: Options):
        self.opts = opts
        apply_migrations(Path(opts.db_path))
        self.services, self.memory = _services(opts)
        self.engine = Engine(Path(opts.db_path), self.services, mode=opts.mode, models=opts.models)

    async def __aenter__(self) -> "Session":
        if isinstance(self.services.browser, McpBrowser):
            await self.services.browser.open()  # same task as close (anyio scopes)
        return self

    async def __aexit__(self, *exc) -> None:
        try:
            await self.memory.close()
        finally:
            if isinstance(self.services.browser, McpBrowser):
                await self.services.browser.close()

    async def run_target(self, target: Dict[str, Any], run_id: Optional[str] = None) -> RunOutcome:
        run_id, graph_id = self.engine.start_run(target, run_id=run_id)
        return await self._drive_new(run_id, graph_id)

    async def run_reflect(self, target: Dict[str, Any]) -> RunOutcome:
        """Memory consolidation for one target (the scheduler's weekly policy)."""
        run_id, graph_id = self.engine.start_run(target, kind="reflect")
        return await self._drive_new(run_id, graph_id)

    async def _drive_new(self, run_id: str, graph_id: str) -> RunOutcome:
        try:
            await self.memory.open(run_id)  # AFTER the runs row is committed
        except BaseException:
            # start_run took the run's lease for this live process. If the
            # memory server cannot start, nothing would ever release it and
            # the run could never be resumed (Phase F review M1, delta M-B).
            # Free it: the graph stays 'running' with no owner, which
            # recovery picks up.
            with db.connect(Path(self.opts.db_path)) as conn:
                lease.release(conn, run_id, self.engine.owner)
            raise
        try:
            return await self.engine.run_graph(graph_id)
        finally:
            await self.memory.close()

    async def resume(self, run_id: str) -> List[RunOutcome]:
        await self.memory.open(run_id)  # fresh server on every resume
        try:
            return await self.engine.resume_run(run_id)
        finally:
            await self.memory.close()


async def watch_all(opts: Options, target_ids: Optional[List[str]] = None) -> List[RunOutcome]:
    """One run per target. A target that fails does not stop the others."""
    outcomes = []
    async with Session(opts) as s:
        for t in config.targets():
            if target_ids and t["id"] not in target_ids:
                continue
            try:
                outcomes.append(await s.run_target(t))
            except Exception as exc:  # noqa: BLE001 - one target never sinks the batch
                outcomes.append(RunOutcome(run_id="", graph_id="", status=f"error: {exc!r}"[:300]))
    return outcomes


def decided_waiting_runs(db_path: Path) -> List[str]:
    """Runs with a graph waiting on approvals that a human has now decided."""
    with db.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT g.run_id FROM graphs g JOIN graph_nodes n ON n.graph_id = g.graph_id "
            "JOIN approvals a ON a.idempotency_key = n.idempotency_key "
            "WHERE g.status = 'waiting_approval' AND n.status = 'blocked' "
            "AND a.status IN ('approved', 'rejected')").fetchall()
    return [r["run_id"] for r in rows]


async def resume_decided(opts: Options) -> List[RunOutcome]:
    outcomes: List[RunOutcome] = []
    async with Session(opts) as s:
        for run_id in decided_waiting_runs(Path(opts.db_path)):
            try:
                outcomes.extend(await s.resume(run_id))
            except lease.LeaseLost:
                outcomes.append(RunOutcome(run_id, "", "owned by a live process (left alone)"))
            except Exception as exc:  # noqa: BLE001 - one run never blocks the others
                outcomes.append(RunOutcome(run_id, "", f"error: {exc!r}"[:300]))
    return outcomes


class DecisionError(ValueError):
    pass


def decide(db_path: Path, approval_id: int, decision: str, *, by: str, note: str = "") -> Dict[str, Any]:
    """The human's half of the gate. Only a PENDING row can be decided, once."""
    if decision not in ("approved", "rejected"):
        raise DecisionError("decision must be 'approved' or 'rejected'")
    with db.connect(db_path) as conn:
        with db.atomic(conn):
            cur = conn.execute(
                "UPDATE approvals SET status = ?, decided_by = ?, decision_note = ?, "
                "resolved_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
                "WHERE approval_id = ? AND status = 'pending'",
                (decision, by, note or None, approval_id))
            if cur.rowcount != 1:
                row = conn.execute("SELECT status FROM approvals WHERE approval_id = ?", (approval_id,)).fetchone()
                raise DecisionError(f"approval {approval_id} is "
                                    f"{row['status'] if row else 'missing'}; only pending approvals can be decided")
        return dict(conn.execute("SELECT * FROM approvals WHERE approval_id = ?", (approval_id,)).fetchone())


def list_approvals(db_path: Path, status: Optional[str] = "pending") -> List[Dict[str, Any]]:
    with db.connect(db_path) as conn:
        sql = "SELECT * FROM approvals"
        params: tuple = ()
        if status:
            sql += " WHERE status = ?"
            params = (status,)
        return [dict(r) for r in conn.execute(sql + " ORDER BY created_at DESC", params)]
