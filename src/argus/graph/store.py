"""Graph persistence. The ONLY module that writes graph state.

Every method that writes opens exactly one `db.atomic()` and returns. No
method awaits anything, so no transaction can be held across a model, MCP or
browser call (CLAUDE.md). The engine computes first, then calls one of these.

Checkpoint rule (ARGUS_BUILD 6.8): a node's status change, its output, and
its `node_events` row land in ONE transaction (`transition`). Expansions are
stored, validated, before any of their nodes can become ready
(`apply_expansion`). Routing decisions are stored before their call runs and
frozen in the same transaction as ready -> running (`start_node`).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from argus.graph import nodes as node_types
from argus.graph import validator
from argus.graph.model import EdgeSpec, GraphSpec, Limits, NodeSpec
from argus.storage import db, lease

#: Legal status changes. Anything else is a bug and raises.
TRANSITIONS = {
    "pending": {"ready", "skipped"},
    "ready": {"running", "skipped", "pending"},
    "running": {"done", "failed", "blocked", "ready", "skipped"},
    "blocked": {"done", "skipped", "failed", "ready"},
    "done": set(),
    "failed": set(),
    "skipped": set(),
}

Op = Tuple[str, Tuple[Any, ...]]


class IllegalTransition(RuntimeError):
    pass


@dataclass
class NodeRow:
    node_id: str
    graph_id: str
    node_type: str
    status: str
    params: Dict[str, Any]
    attempt: int
    retry_of: Optional[str]
    retry_root: str
    idempotency_key: Optional[str]
    prunable: bool
    difficulty: float
    template_instance: Optional[str]
    output: Optional[Dict[str, Any]]
    error: Optional[str]
    skip_reason: Optional[str]
    created_at: str
    started_at: Optional[str]
    finished_at: Optional[str]

    def spec(self) -> NodeSpec:
        return NodeSpec(
            node_id=self.node_id, node_type=self.node_type, params=dict(self.params),
            attempt=self.attempt, retry_of=self.retry_of, retry_root=self.retry_root,
            idempotency_key=self.idempotency_key, prunable=self.prunable,
            difficulty=self.difficulty, status=self.status,
            template_instance=self.template_instance,
        )


@dataclass
class GraphRow:
    graph_id: str
    run_id: str
    target_id: str
    kind: str
    status: str
    limits: Limits
    config: Dict[str, Any]
    budget_s: Optional[float]
    budget_tokens: Optional[int]
    expansions_used: int
    tokens_used: int
    seconds_used: float


def _now() -> str:
    from argus.orchestrator.context import utcnow_iso

    return utcnow_iso()


def _node_row(r: sqlite3.Row) -> NodeRow:
    return NodeRow(
        node_id=r["node_id"], graph_id=r["graph_id"], node_type=r["node_type"],
        status=r["status"], params=json.loads(r["params"] or "{}"), attempt=r["attempt"],
        retry_of=r["retry_of"], retry_root=r["retry_root"],
        idempotency_key=r["idempotency_key"], prunable=bool(r["prunable"]),
        difficulty=r["difficulty"], template_instance=r["template_instance"],
        output=json.loads(r["output"]) if r["output"] is not None else None,
        error=r["error"], skip_reason=r["skip_reason"], created_at=r["created_at"],
        started_at=r["started_at"], finished_at=r["finished_at"],
    )


def _default_prunable(node: NodeSpec) -> bool:
    nt = node_types.get(node.node_type)
    if node.prunable is not None:
        return bool(node.prunable)
    return bool(nt.prunable and not nt.side_effect)


class GraphStore:
    def __init__(self, conn: sqlite3.Connection, *, run_id: Optional[str] = None,
                 lease_owner: Optional[str] = None):
        """With `lease_owner`, EVERY write transaction first proves this
        process still owns the run (heartbeat UPDATE + rowcount check, audit
        H2). A process that lost the run cannot write one row of graph state."""
        self.conn = conn
        self.run_id = run_id
        self.lease_owner = lease_owner

    @contextmanager
    def _tx(self):
        with db.atomic(self.conn):
            if self.lease_owner is not None:
                lease.assert_held(self.conn, self.run_id, self.lease_owner)
            yield

    # ------------------------------------------------------------------ reads

    def graph(self, graph_id: str) -> GraphRow:
        r = self.conn.execute("SELECT * FROM graphs WHERE graph_id = ?", (graph_id,)).fetchone()
        if r is None:
            raise KeyError(graph_id)
        return GraphRow(
            graph_id=r["graph_id"], run_id=r["run_id"], target_id=r["target_id"],
            kind=r["kind"], status=r["status"], limits=Limits.from_dict(json.loads(r["limits"])),
            config=json.loads(r["config"] or "{}"), budget_s=r["budget_s"],
            budget_tokens=r["budget_tokens"], expansions_used=r["expansions_used"],
            tokens_used=r["tokens_used"], seconds_used=r["seconds_used"],
        )

    def nodes(self, graph_id: str) -> Dict[str, NodeRow]:
        rows = self.conn.execute(
            "SELECT * FROM graph_nodes WHERE graph_id = ? ORDER BY created_at, node_id",
            (graph_id,),
        ).fetchall()
        return {r["node_id"]: _node_row(r) for r in rows}

    def node(self, node_id: str) -> NodeRow:
        r = self.conn.execute("SELECT * FROM graph_nodes WHERE node_id = ?", (node_id,)).fetchone()
        if r is None:
            raise KeyError(node_id)
        return _node_row(r)

    def edges(self, graph_id: str) -> List[EdgeSpec]:
        return [
            EdgeSpec(r["src"], r["dst"], r["channel"])
            for r in self.conn.execute(
                "SELECT src, dst, channel FROM graph_edges WHERE graph_id = ? ORDER BY rowid",
                (graph_id,),
            )
        ]

    def spec(self, graph_id: str) -> GraphSpec:
        return GraphSpec.build((n.spec() for n in self.nodes(graph_id).values()), self.edges(graph_id))

    def events(self, node_id: str) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM node_events WHERE node_id = ? ORDER BY seq", (node_id,)
        ).fetchall()

    def live_decision(self, node_id: str) -> Optional[Dict[str, Any]]:
        r = self.conn.execute(
            "SELECT * FROM routing_decisions WHERE node_id = ? AND superseded = 0", (node_id,)
        ).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["models"] = json.loads(d["models"])
        d["candidates"] = json.loads(d["candidates"])
        return d

    def samples(self, node_id: str) -> Dict[int, Dict[str, Any]]:
        out = {}
        for r in self.conn.execute(
            "SELECT * FROM node_samples WHERE node_id = ? ORDER BY sample_index", (node_id,)
        ):
            out[r["sample_index"]] = {
                "model_tag": r["model_tag"], "ok": bool(r["ok"]), "error": r["error"],
                "output": json.loads(r["output"]) if r["output"] is not None else None,
            }
        return out

    # ----------------------------------------------------------------- writes

    def create_graph(
        self,
        *,
        run_id: str,
        target_id: str,
        kind: str,
        limits: Limits,
        config: Optional[Dict[str, Any]] = None,
        budget_s: Optional[float] = None,
        budget_tokens: Optional[int] = None,
        graph_id: Optional[str] = None,
    ) -> str:
        graph_id = graph_id or f"graph-{uuid.uuid4().hex[:12]}"
        with self._tx():
            self.conn.execute(
                "INSERT INTO graphs (graph_id, run_id, target_id, kind, status, limits, config, "
                "budget_s, budget_tokens) VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?)",
                (graph_id, run_id, target_id, kind, json.dumps(limits.to_dict()),
                 json.dumps(config or {}, sort_keys=True), budget_s, budget_tokens),
            )
        return graph_id

    def apply_expansion(
        self,
        graph_id: str,
        expansion: GraphSpec,
        *,
        kind: str,
        requested_by: Optional[str] = None,
        expansion_id: Optional[str] = None,
        counts_toward_limit: bool = True,
        allow_existing_dst: bool = False,
        complete_node: Optional[Tuple[str, Dict[str, Any]]] = None,
        extra_ops: Sequence[Op] = (),
    ) -> Tuple[bool, List[validator.Violation]]:
        """Validate, then store the expansion and its nodes in ONE transaction.

        The expansion row is written whether or not it is accepted, so a
        rejected planner output is visible in the trace. Idempotent by
        `expansion_id`: re-applying a stored expansion (a resumed replan)
        returns the stored verdict and writes nothing.
        """
        expansion_id = expansion_id or f"exp-{uuid.uuid4().hex[:12]}"
        prior = self.conn.execute(
            "SELECT accepted, violations FROM graph_expansions WHERE expansion_id = ?",
            (expansion_id,),
        ).fetchone()
        if prior is not None:
            if complete_node is not None and self.node(complete_node[0]).status == "running":
                # The node and its expansion always commit together, so a
                # stored expansion with a still-running requester means the
                # id was reused by someone else: a bug, never a resume.
                raise RuntimeError(f"expansion id {expansion_id!r} already used by another expansion")
            return bool(prior["accepted"]), [
                validator.Violation(**v) for v in json.loads(prior["violations"] or "[]")
            ]

        g = self.graph(graph_id)
        existing = self.spec(graph_id)
        used = g.expansions_used + (1 if counts_toward_limit else 0)
        violations = validator.validate_expansion(
            existing, expansion, g.limits, used, allow_existing_dst=allow_existing_dst
        )
        accepted = not violations
        spec_json = json.dumps({
            "nodes": [
                {"node_id": n.node_id, "node_type": n.node_type, "params": n.params,
                 "attempt": n.attempt, "retry_of": n.retry_of, "retry_root": n.retry_root,
                 "template_instance": n.template_instance}
                for n in expansion.nodes.values()
            ],
            "edges": [[e.src, e.dst, e.channel] for e in expansion.edges],
        }, sort_keys=True, default=str)
        with self._tx():
            self.conn.execute(
                "INSERT INTO graph_expansions (expansion_id, graph_id, requested_by, kind, spec, "
                "accepted, violations) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (expansion_id, graph_id, requested_by, kind, spec_json, int(accepted),
                 json.dumps([v.__dict__ for v in violations])),
            )
            if complete_node is not None:
                # The planner node's output and the nodes it created land
                # together: a crash can never leave a done planner without
                # its expansion, or an expansion whose planner re-runs.
                node_id, output = complete_node
                self._set_status(node_id, "done", expect=("running",),
                                 output={**output, "expansion_id": expansion_id,
                                         "expansion_accepted": accepted,
                                         "violations": [str(v) for v in violations]})
                for sql, params in extra_ops:  # e.g. the planner's rewards
                    self.conn.execute(sql, params)
            if not accepted:
                return False, violations
            for n in expansion.nodes.values():
                self.conn.execute(
                    "INSERT INTO graph_nodes (node_id, graph_id, node_type, status, params, attempt, "
                    "retry_of, retry_root, idempotency_key, prunable, difficulty, template_instance, "
                    "expansion_id) VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (n.node_id, graph_id, n.node_type, json.dumps(n.params, sort_keys=True, default=str),
                     n.attempt, n.retry_of, n.retry_root, n.idempotency_key,
                     int(_default_prunable(n)), n.difficulty, n.template_instance, expansion_id),
                )
                self._event(graph_id, n.node_id, None, "pending", {"expansion": expansion_id})
            for e in expansion.edges:
                self.conn.execute(
                    "INSERT INTO graph_edges (graph_id, src, dst, channel) VALUES (?, ?, ?, ?)",
                    (graph_id, e.src, e.dst, e.channel),
                )
            if counts_toward_limit:
                self.conn.execute(
                    "UPDATE graphs SET expansions_used = expansions_used + 1 WHERE graph_id = ?",
                    (graph_id,),
                )
        return True, []

    def _event(self, graph_id: str, node_id: str, frm: Optional[str], to: str,
               detail: Optional[Dict[str, Any]] = None) -> None:
        seq = self.conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM node_events WHERE node_id = ?", (node_id,)
        ).fetchone()[0]
        self.conn.execute(
            "INSERT INTO node_events (graph_id, node_id, seq, from_status, to_status, detail) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (graph_id, node_id, seq, frm, to, json.dumps(detail, sort_keys=True, default=str) if detail else None),
        )

    def _set_status(
        self,
        node_id: str,
        to: str,
        *,
        expect: Optional[Iterable[str]] = None,
        output: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        skip_reason: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> str:
        row = self.conn.execute(
            "SELECT graph_id, status FROM graph_nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        if row is None:
            raise KeyError(node_id)
        frm = row["status"]
        if expect is not None and frm not in set(expect):
            raise IllegalTransition(f"{node_id}: expected {sorted(expect)}, found {frm}")
        if to not in TRANSITIONS[frm]:
            raise IllegalTransition(f"{node_id}: {frm} -> {to} is not allowed")
        now = _now()
        sets = ["status = ?"]
        params: List[Any] = [to]
        if to == "running":
            sets.append("started_at = ?")
            params.append(now)
        if to in ("done", "failed", "skipped"):
            sets.append("finished_at = ?")
            params.append(now)
        if output is not None:
            sets.append("output = ?")
            params.append(json.dumps(output, sort_keys=True, default=str))
        if error is not None:
            sets.append("error = ?")
            params.append(error)
        if skip_reason is not None:
            sets.append("skip_reason = ?")
            params.append(skip_reason)
        params.append(node_id)
        self.conn.execute(f"UPDATE graph_nodes SET {', '.join(sets)} WHERE node_id = ?", params)  # security-scan: ok column names are literals above, values bound
        self._event(row["graph_id"], node_id, frm, to, detail)
        return frm

    def transition(
        self,
        node_id: str,
        to: str,
        *,
        expect: Optional[Iterable[str]] = None,
        output: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        skip_reason: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
        extra_ops: Sequence[Op] = (),
    ) -> None:
        """One node status change + its checkpoint event (+ extra ops), atomically."""
        if to == "done" and output is None:
            raise ValueError("a node can only become done with its output")
        with self._tx():
            self._set_status(node_id, to, expect=expect, output=output, error=error,
                             skip_reason=skip_reason, detail=detail)
            for sql, params in extra_ops:
                self.conn.execute(sql, params)

    def transition_many(self, changes: Sequence[Tuple[str, str, Dict[str, Any]]]) -> None:
        """Several (node_id, to, kwargs) changes in one transaction (readiness sweeps)."""
        if not changes:
            return
        with self._tx():
            for node_id, to, kwargs in changes:
                self._set_status(node_id, to, **kwargs)

    # -------------------------------------------------------------- routing

    def store_decision(
        self,
        *,
        graph_id: str,
        node_id: str,
        node_type: str,
        model_tag: str,
        models: List[str],
        strategy: str,
        candidates: List[Dict[str, Any]],
        mode: str,
        explore: bool,
        reason: str,
        revise: bool = False,
    ) -> str:
        """Store a live routing decision. With revise=True, supersede the
        current live one (only if it is not frozen) and link to it."""
        decision_id = f"dec-{uuid.uuid4().hex[:12]}"
        with self._tx():
            live = self.conn.execute(
                "SELECT decision_id, frozen FROM routing_decisions WHERE node_id = ? AND superseded = 0",
                (node_id,),
            ).fetchone()
            revised_from = None
            if live is not None:
                if not revise:
                    return live["decision_id"]
                if live["frozen"]:
                    raise IllegalTransition(f"{node_id}: decision is frozen, cannot revise")
                self.conn.execute(
                    "UPDATE routing_decisions SET superseded = 1 WHERE decision_id = ?",
                    (live["decision_id"],),
                )
                revised_from = live["decision_id"]
            self.conn.execute(
                "INSERT INTO routing_decisions (decision_id, graph_id, node_id, node_type, model_tag, "
                "models, strategy, candidates, mode, explore, revised_from, reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (decision_id, graph_id, node_id, node_type, model_tag, json.dumps(models), strategy,
                 json.dumps(candidates, sort_keys=True, default=str), mode, int(explore),
                 revised_from, reason),
            )
        return decision_id

    def start_node(self, node_id: str) -> None:
        """ready -> running, freezing the live routing decision in the same tx."""
        with self._tx():
            self._set_status(node_id, "running", expect=("ready",))
            self.conn.execute(
                "UPDATE routing_decisions SET frozen = 1 WHERE node_id = ? AND superseded = 0",
                (node_id,),
            )

    def store_sample(self, node_id: str, index: int, model_tag: str,
                     output: Optional[Dict[str, Any]], ok: bool, error: Optional[str]) -> None:
        with self._tx():
            self.conn.execute(
                "INSERT INTO node_samples (node_id, sample_index, model_tag, output, ok, error) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (node_id, sample_index) DO NOTHING",
                (node_id, index, model_tag,
                 json.dumps(output, sort_keys=True, default=str) if output is not None else None,
                 int(ok), error),
            )

    # ---------------------------------------------------------------- graph

    def set_graph_status(self, graph_id: str, status: str) -> None:
        with self._tx():
            finished = _now() if status in ("done", "failed") else None
            self.conn.execute(
                "UPDATE graphs SET status = ?, finished_at = COALESCE(?, finished_at) "
                "WHERE graph_id = ?",
                (status, finished, graph_id),
            )

    def finish_graph(self, graph_id: str, status: str) -> None:
        """Graph status AND its run's status, one transaction (review M5).
        The run is closed only when none of its graphs is still open."""
        with self._tx():
            finished = _now() if status in ("done", "failed") else None
            self.conn.execute(
                "UPDATE graphs SET status = ?, finished_at = COALESCE(?, finished_at) WHERE graph_id = ?",
                (status, finished, graph_id))
            if status in ("done", "failed"):
                run_id = self.conn.execute("SELECT run_id FROM graphs WHERE graph_id = ?",
                                           (graph_id,)).fetchone()["run_id"]
                self.conn.execute(
                    "UPDATE runs SET status = ?, finished_at = ? WHERE run_id = ? AND NOT EXISTS "
                    "(SELECT 1 FROM graphs WHERE run_id = ? AND status IN ('running', 'waiting_approval'))",
                    (status, finished, run_id, run_id))

    def add_usage(self, graph_id: str, *, tokens: int = 0, seconds: float = 0.0) -> None:
        with self._tx():
            self.conn.execute(
                "UPDATE graphs SET tokens_used = tokens_used + ?, seconds_used = seconds_used + ? "
                "WHERE graph_id = ?",
                (int(tokens), float(seconds), graph_id),
            )

    def reset_running_for_resume(self, graph_id: str) -> List[str]:
        """Crash recovery: every running node goes back to ready, one tx.

        Its routing decision stays frozen and is reused. Its stored samples
        stay and are reused. Nothing else is recomputed.
        """
        ids = [
            r["node_id"] for r in self.conn.execute(
                "SELECT node_id FROM graph_nodes WHERE graph_id = ? AND status = 'running'",
                (graph_id,),
            )
        ]
        if ids:
            with self._tx():
                for node_id in ids:
                    self._set_status(node_id, "ready", expect=("running",),
                                     detail={"reason": "resume"})
        return ids
