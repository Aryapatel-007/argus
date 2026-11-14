"""The graph engine: scheduler + routing + checkpointed execution.

Two lanes (ARGUS_BUILD 6.7):
  - GPU lane: ONE LLM node in flight at a time, and every model call inside
    it holds `gpu` (asyncio.Lock) for its duration.
  - I/O lane: up to `io_slots` tool/pure nodes concurrently (page fetches,
    memory calls), so a page load overlaps a model call.

Each loop iteration works only from stored state:
  1. heartbeat the run lease (a process that lost the run stops)
  2. plan retries for failed nodes (deterministic ids; idempotent)
  3. add a replan node for every node that emitted a trigger
  4. resolve blocked approval gates from the approvals table
  5. prune optional work if the budget is nearly spent
  6. sweep: pending -> ready / skipped
  7. route newly ready LLM nodes (decision STORED before the call)
  8. dispatch: I/O nodes up to the slot limit; one LLM node, chosen
     swap-aware (resident model first, revise to resident if cheap)
  9. wait for the first node to finish and store its result

Crash-resume: on start, every `running` node goes back to `ready`, keeping its
frozen routing decision and stored samples; nothing done is recomputed.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from argus import config
from argus.graph import bandit, handlers, procedural, readiness, templates, validator
from argus.graph import nodes as node_types
from argus.graph.health import HealthMonitor
from argus.graph.llm import CallResult
from argus.graph.model import EdgeSpec, GraphSpec, Limits, NodeSpec, new_node_id
from argus.graph.router import GraphRoutingState, Router, load_routing_config
from argus.graph.services import Services
from argus.graph.store import GraphStore, NodeRow
from argus.storage import db, lease

LLM_KINDS = {"llm"}


@dataclass
class NodeExecution:
    """Everything a handler may see. Built per node run."""

    store: GraphStore
    node: NodeRow
    inputs: Dict[str, Any]
    params: Dict[str, Any]
    target: Dict[str, Any]
    graph_id: str
    run_id: str
    role: str
    services: Services
    gpu: asyncio.Lock
    routing_cfg: Dict[str, Any]
    decision: Optional[Dict[str, Any]]
    on_model_used: Callable[[str, CallResult], None]
    pick_second_model: Callable[..., Optional[str]]


@dataclass
class RunOutcome:
    run_id: str
    graph_id: str
    status: str
    digest: Optional[str] = None
    nodes: Dict[str, int] = field(default_factory=dict)


class Engine:
    def __init__(self, db_path: Path, services: Services, *, mode: str = "fixed",
                 routing_cfg: Optional[Dict[str, Any]] = None, models: Optional[List[str]] = None,
                 baseline: Optional[str] = None, owner: Optional[str] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.db_path = Path(db_path)
        self.services = services
        self.mode = mode
        self.cfg = routing_cfg if routing_cfg is not None else load_routing_config()
        self.models = [config.normalize_tag(m) for m in (models or config.allowed_models())]
        self.baseline = baseline
        self.owner = owner or lease.new_owner()
        self.clock = clock
        self.gpu = asyncio.Lock()
        self.resident: Set[str] = set()

    # ------------------------------------------------------------ start

    def start_run(self, target: Dict[str, Any], *, kind: str = "watch",
                  run_id: Optional[str] = None, conn=None) -> Tuple[str, str]:
        """Create the run (+ lease, one tx), its graph, and the initial plan
        node. Returns (run_id, graph_id)."""
        run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        own = conn is None
        cm = db.connect(self.db_path) if own else None
        conn = cm.__enter__() if own else conn
        try:
            with db.atomic(conn):
                conn.execute(
                    "INSERT INTO runs (run_id, target_id, status) VALUES (?, ?, 'running')",
                    (run_id, target["id"]),
                )
                lease.insert_initial(conn, run_id, self.owner)
        except BaseException:
            if own:
                cm.__exit__(None, None, None)
            raise
        try:
            store = GraphStore(conn, run_id=run_id, lease_owner=self.owner)
            limits = Limits.from_dict(self.cfg.get("limits"))
            budget = self.cfg.get("budget", {})
            graph_id = store.create_graph(
                run_id=run_id, target_id=target["id"], kind=kind, limits=limits,
                config={"target": target, "mode": self.mode},
                budget_s=budget.get("seconds"), budget_tokens=budget.get("tokens"),
            )
            if kind == "reflect":
                initial, _ = templates.instantiate("reflect", graph_id, target["id"], {})
            else:
                initial = GraphSpec.build([NodeSpec(
                    node_id=new_node_id("plan"), node_type="plan",
                    params={"fields": list(target.get("extract") or []),
                            # stored with the node, so a resumed plan sees the
                            # same hints it was created with
                            "hints": procedural.hints_for(conn, target["id"])},
                    prunable=False)], [])
            ok, violations = store.apply_expansion(graph_id, initial, kind="initial",
                                                   counts_toward_limit=False)
            if not ok:
                raise RuntimeError(f"initial graph invalid: {violations}")
            return run_id, graph_id
        except BaseException as exc:
            # The runs row and its lease are already committed. Never leave a
            # live process holding a run nobody will drive (delta review
            # M-B): close the run as failed, then free it.
            with db.atomic(conn):
                conn.execute("UPDATE runs SET status = 'failed', error = ?, "
                             "finished_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE run_id = ?",
                             (f"start failed: {exc!r}"[:500], run_id))
                conn.execute("UPDATE graphs SET status = 'failed' WHERE run_id = ?", (run_id,))
            lease.release(conn, run_id, self.owner)
            raise
        finally:
            if own:
                cm.__exit__(None, None, None)

    # -------------------------------------------------------------- run

    async def run_graph(self, graph_id: str, *, release: bool = True) -> RunOutcome:
        with db.connect(self.db_path) as conn:
            base = GraphStore(conn)
            g = base.graph(graph_id)
            # Leases are released when a graph finishes or waits, so (re)take
            # it here; a live owner elsewhere makes this refuse.
            if not lease.acquire(conn, g.run_id, self.owner):
                raise lease.LeaseLost(f"run {g.run_id} is owned by a live process")
            store = GraphStore(conn, run_id=g.run_id, lease_owner=self.owner)
            try:
                return await self._drive(store, g)
            finally:
                # Every exit, including an exception, frees the run so another
                # engine in this same live process can resume it (delta review 6).
                if release:
                    lease.release(conn, g.run_id, self.owner)

    async def resume_run(self, run_id: str) -> List[RunOutcome]:
        """Take over a run (lease) and drive every unfinished graph of it."""
        outcomes = []
        with db.connect(self.db_path) as conn:
            if not lease.acquire(conn, run_id, self.owner):
                raise lease.LeaseLost(f"run {run_id} is owned by a live process")
            ids = [r["graph_id"] for r in conn.execute(
                "SELECT graph_id FROM graphs WHERE run_id = ? AND status IN ('running', 'waiting_approval') "
                "ORDER BY created_at", (run_id,))]
        try:
            for gid in ids:
                # Hold the lease ACROSS graphs: releasing between them would let
                # another process take the run mid-resume (delta review 7).
                outcomes.append(await self.run_graph(gid, release=False))
        finally:
            with db.connect(self.db_path) as conn:
                lease.release(conn, run_id, self.owner)
        return outcomes

    async def _drive(self, store: GraphStore, g) -> RunOutcome:
        target = g.config.get("target") or {"id": g.target_id}
        router = Router(store.conn, self.models, self.mode, self.cfg, self.baseline)
        health = router.health  # one monitor: trial grants and results meet here
        state = self._routing_state(store, g.graph_id)
        if store.graph(g.graph_id).status == "waiting_approval":
            store.set_graph_status(g.graph_id, "running")
        store.reset_running_for_resume(g.graph_id)
        self.resident = set(await self.services.backend.resident_models())
        io_sem = asyncio.Semaphore(int(self.cfg.get("io_slots", 4)))
        tasks: Dict[asyncio.Task, str] = {}
        llm_in_flight: Optional[str] = None
        last_tick = self.clock()

        def on_model_used(model: str, result: CallResult) -> None:
            self.resident = {model} | {m for m in self.resident if "embed" in m}
            health.record(model, tok_s=result.tok_s, timed_out=result.timed_out, ok=result.ok,
                          responded=result.raw is not None)

        while True:
            # Yield every iteration: a loop that only writes and `continue`s
            # would otherwise starve every task and every timeout (review
            # High 1 could not even be cancelled).
            await asyncio.sleep(0)
            now = self.clock()
            store.add_usage(g.graph_id, seconds=now - last_tick)  # also heartbeats the lease
            last_tick = now
            g = store.graph(g.graph_id)
            nodes, edges = store.nodes(g.graph_id), store.edges(g.graph_id)
            running = set(tasks.values())

            if self._housekeeping(store, g, target, nodes, edges, state):
                continue

            # route every ready LLM node that has no live decision yet
            for n in nodes.values():
                if n.status == "ready" and self._kind(n) in LLM_KINDS and n.node_id not in running:
                    if store.live_decision(n.node_id) is None:
                        self._route(store, router, state, g, n, nodes, edges)

            ready = [n for n in nodes.values() if n.status == "ready" and n.node_id not in running]
            for n in ready:
                if self._kind(n) not in LLM_KINDS and len([t for t in tasks.values()
                                                           if self._kind(nodes[t]) not in LLM_KINDS]) < int(self.cfg.get("io_slots", 4)):
                    tasks[asyncio.create_task(self._run_node(store, g, target, n, nodes, edges,
                                                             on_model_used, router, io_sem))] = n.node_id
            if llm_in_flight is None:
                llm_ready = [n for n in ready if self._kind(n) in LLM_KINDS]
                pick = self._pick_llm(store, router, llm_ready, nodes, edges) if llm_ready else None
                if pick is not None:
                    llm_in_flight = pick.node_id
                    tasks[asyncio.create_task(self._run_node(store, g, target, pick, nodes, edges,
                                                             on_model_used, router, None))] = pick.node_id

            if not tasks:
                outcome = readiness.graph_outcome(store.nodes(g.graph_id))
                if outcome is None:
                    # Nothing runnable but not finished: a sweep will move it on.
                    sweep = readiness.sweep(store.nodes(g.graph_id), store.edges(g.graph_id))
                    if not sweep.ready and not sweep.skip:
                        outcome = "failed"  # stuck: should be unreachable; never spin
                    else:
                        continue
                if outcome in ("done", "failed"):
                    # Idempotent (one row per instance), so recording before
                    # the finish cannot double count after a crash; recording
                    # at waiting_approval would freeze notify as failed.
                    self._record_templates(store, g.graph_id)
                store.finish_graph(g.graph_id, outcome)
                return self._outcome(store, g, outcome)

            done, _ = await asyncio.wait(list(tasks), return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                node_id = tasks.pop(t)
                if node_id == llm_in_flight:
                    llm_in_flight = None
                result = t.result()
                self._store_result(store, g, target, node_id, result)

    # ---------------------------------------------------------- helpers

    @staticmethod
    def _kind(n) -> str:
        return node_types.get(n.node_type).kind

    def _routing_state(self, store: GraphStore, graph_id: str) -> GraphRoutingState:
        row = store.conn.execute(
            "SELECT COUNT(*) AS d, COALESCE(SUM(explore), 0) AS e FROM routing_decisions "
            "WHERE graph_id = ? AND superseded = 0", (graph_id,)).fetchone()
        return GraphRoutingState(decisions=row["d"], explorations=row["e"])

    def _budget_remaining(self, g) -> float:
        fr = []
        if g.budget_s:
            fr.append(1.0 - g.seconds_used / g.budget_s)
        if g.budget_tokens:
            fr.append(1.0 - g.tokens_used / g.budget_tokens)
        return max(0.0, min(fr)) if fr else 1.0

    def _housekeeping(self, store, g, target, nodes, edges, state) -> bool:
        """Steps 2-6. True if anything was written (caller reloads)."""
        changed = False
        # 2. retries (state-derived, deterministic ids)
        # A retry the validator rejected (node or depth limit) is FINAL: it is
        # recorded once and never re-offered, or this loop would spin forever
        # without awaiting (review High 1). The failed chain then ends dead.
        rejected = {r["expansion_id"] for r in store.conn.execute(
            "SELECT expansion_id FROM graph_expansions WHERE graph_id = ? AND accepted = 0",
            (g.graph_id,))}
        for exp_id, spec in readiness.plan_retries(nodes, edges, g.limits):
            if exp_id in rejected:
                continue
            ok, _ = store.apply_expansion(g.graph_id, spec, kind="retry", expansion_id=exp_id,
                                          counts_toward_limit=False, allow_existing_dst=True)
            changed = True  # either nodes were added or a rejection was recorded
        # 3. triggers -> replan nodes
        for n in nodes.values():
            if n.status == "done" and (n.output or {}).get("trigger"):
                rid = f"replan-{n.node_id}"
                if rid in nodes or g.expansions_used >= g.limits.max_expansions:
                    continue
                if n.node_type == "extract" and not any(
                        e.src == n.node_id and e.channel == "subject" and nodes[e.dst].status == "done"
                        for e in edges if e.dst in nodes):
                    continue  # never replan from facts the critic has not accepted
                spec = GraphSpec.build(
                    [NodeSpec(node_id=rid, node_type="replan",
                              params={"fields": n.output["trigger"].get("missing_fields", [])})],
                    [EdgeSpec(n.node_id, rid, "trigger")])
                # "trigger-" prefix: "exp-<node_id>" is reserved for a planner
                # node's OWN output expansion. Sharing it made the replan's
                # expansion look already applied and stranded the node.
                ok, _ = store.apply_expansion(g.graph_id, spec, kind="engine", expansion_id=f"trigger-{rid}",
                                              counts_toward_limit=False)
                changed = changed or ok
        # 4. blocked gates
        for n in nodes.values():
            if n.status != "blocked":
                continue
            row = store.conn.execute(
                "SELECT approval_id, status, decided_by, decision_note FROM approvals WHERE idempotency_key = ?",
                (n.idempotency_key,)).fetchone()
            if row is None:
                continue
            if row["status"] in ("approved", "executed"):
                store.transition(n.node_id, "done", expect=("blocked",), output={"approval": {
                    "idempotency_key": n.idempotency_key, "approval_id": row["approval_id"],
                    "status": "approved", "decided_by": row["decided_by"]}})
                changed = True
            elif row["status"] in ("rejected", "failed"):
                store.transition(n.node_id, "skipped", expect=("blocked",),
                                 skip_reason=f"approval {row['status']}: {row['decision_note'] or ''}".strip())
                changed = True
        # 5. budget pruning
        remaining = self._budget_remaining(g)
        state.budget_remaining = remaining
        if remaining < float(self.cfg.get("budget", {}).get("prune_fraction", 0.15)):
            prunable = [(n.node_id, "skipped", {"skip_reason": "budget"})
                        for n in nodes.values() if n.status in ("pending", "ready") and n.prunable
                        and not node_types.get(n.node_type).side_effect]
            if prunable:
                store.transition_many(prunable)
                changed = True
        # 6. sweep
        if changed:
            return True
        sweep = readiness.sweep(nodes, edges)
        moves = [(nid, "ready", {"expect": ("pending",)}) for nid in sweep.ready]
        moves += [(nid, "skipped", {"expect": ("pending",), "skip_reason": why}) for nid, why in sweep.skip.items()]
        if moves:
            store.transition_many(moves)
            return True
        return False

    def _exclusions(self, store, n, nodes, edges) -> Set[str]:
        """A critic never judges its own model's work (ARGUS_BUILD 6.6)."""
        if n.node_type != "critique":
            return set()
        out = set()
        for e in edges:
            if e.dst == n.node_id and e.channel == "subject" and e.src in nodes:
                src = nodes[e.src]
                produced = (src.output or {}).get("_produced_by") if src.status == "done" else None
                if produced:
                    out.add(produced)
                else:
                    d = store.live_decision(e.src)
                    if d:
                        out.update(d["models"])
        return out

    def _route(self, store, router, state, g, n, nodes, edges) -> None:
        d = router.decide(node_id=n.node_id, node_type=n.node_type, difficulty=n.difficulty,
                          attempt=n.attempt, resident=self.resident,
                          exclude=self._exclusions(store, n, nodes, edges), state=state,
                          has_reflection=bool(n.params.get("reflection")),
                          default_strategy=node_types.get(n.node_type).default_strategy)
        store.store_decision(graph_id=g.graph_id, node_id=n.node_id, node_type=n.node_type,
                             model_tag=d.model, models=d.models, strategy=d.strategy,
                             candidates=d.candidates, mode=d.mode, explore=d.explore, reason=d.reason)

    def _pick_llm(self, store, router, ready, nodes, edges):
        """Swap-aware batching: drain the resident model's queue first; else
        try to revise a not-yet-started decision onto the resident model
        (logged as a superseding decision); else the model with most work."""
        decisions = {n.node_id: store.live_decision(n.node_id) for n in ready}
        ordered = sorted(ready, key=lambda n: (n.created_at, n.node_id))
        on_resident = [n for n in ordered if decisions[n.node_id] and decisions[n.node_id]["model_tag"] in self.resident]
        if on_resident:
            return on_resident[0]
        for n in ordered:
            d = decisions[n.node_id]
            if d is None or d["frozen"] or d["strategy"] != "single":
                continue
            rev = router.revise_to_resident(node_type=n.node_type, difficulty=n.difficulty, current=d,
                                            resident=self.resident,
                                            exclude=self._exclusions(store, n, nodes, edges))
            if rev is not None:
                store.store_decision(
                    graph_id=n.graph_id, node_id=n.node_id, node_type=n.node_type,
                    model_tag=rev["model"], models=[rev["model"]], strategy="single",
                    candidates=d["candidates"], mode=d["mode"], explore=False, revise=True,
                    reason=f"revised {d['model_tag']} -> resident {rev['model']} to avoid a swap "
                           f"(utility loss {rev['revision_loss']:.3f})")
                return n
        counts: Dict[str, int] = {}
        for n in ordered:
            m = (decisions[n.node_id] or {}).get("model_tag", "")
            counts[m] = counts.get(m, 0) + 1
        top_model = max(counts, key=lambda m: (counts[m], m))
        return next(n for n in ordered if (decisions[n.node_id] or {}).get("model_tag", "") == top_model)

    async def _run_node(self, store, g, target, n, nodes, edges, on_model_used, router, io_sem):
        try:
            inputs = readiness.resolve_inputs(n.node_id, nodes, edges)
            store.start_node(n.node_id)  # freezes the routing decision in the same tx
            decision = store.live_decision(n.node_id)

            def pick_second(exclude=()):
                if self.mode == "baseline":
                    return None
                for c in router.candidates(n.node_type, n.difficulty, self.resident, exclude):
                    if not c["excluded"] and c["model"] not in set(exclude):
                        return c["model"]
                return None

            ex = NodeExecution(store=store, node=store.node(n.node_id), inputs=inputs,
                               params=dict(n.params), target=target, graph_id=g.graph_id,
                               run_id=g.run_id, role=node_types.get(n.node_type).role or n.node_type,
                               services=self.services, gpu=self.gpu, routing_cfg=self.cfg,
                               decision=decision, on_model_used=on_model_used,
                               pick_second_model=pick_second)
            if io_sem is not None:
                async with io_sem:
                    return await handlers.run(ex)
            return await handlers.run(ex)
        except lease.LeaseLost:
            raise
        except Exception as exc:  # noqa: BLE001 - one node must never kill the graph
            return handlers.NodeResult("failed", error=f"{type(exc).__name__}: {exc}"[:1000])

    def _store_result(self, store, g, target, node_id, r: handlers.NodeResult) -> None:
        node = store.node(node_id)
        if r.tokens:
            store.add_usage(g.graph_id, tokens=r.tokens)
        # Rewards ride in the SAME transaction as the node's result (review M6):
        # a crash can lose neither one without the other.
        reward_ops = self._reward_ops(store, node, r)
        notes = {"notes": r.notes} if r.notes else None
        if node.node_type in ("plan", "replan") and r.status == "done":
            self._apply_plan(store, g, target, node, r.output or {}, extra_ops=reward_ops)
        elif node.node_type == "plan" and r.status == "failed":
            # The planner never sinks a run: fall back to the target's own
            # page and fields (deterministic), and say so in the output.
            fallback = {"thought": "planner failed; deterministic fallback",
                        "steps": [{"template": "check_page",
                                   "params": {"url": target.get("url"), "fields": list(target.get("extract") or [])}}]}
            self._apply_plan(store, g, target, node, {"plan": fallback, "fallback": True,
                                                      "planner_error": r.error}, extra_ops=reward_ops)
        elif node.node_type == "replan" and r.status == "failed":
            store.transition(node_id, "done", expect=("running",), extra_ops=reward_ops,
                             output={"plan": {"thought": "replan failed", "steps": []}, "error": r.error})
        elif r.status == "done":
            store.transition(node_id, "done", expect=("running",), output=r.output or {},
                             detail=notes, extra_ops=list(r.extra_ops) + reward_ops)
        elif r.status == "blocked":
            store.transition(node_id, "blocked", expect=("running",), extra_ops=list(r.extra_ops) + reward_ops,
                             detail=notes)
        elif r.status == "skipped":
            store.transition(node_id, "skipped", expect=("running",), skip_reason=r.skip_reason,
                             extra_ops=reward_ops)
        else:
            store.transition(node_id, "failed", expect=("running",), output=r.output, error=r.error,
                             detail=notes, extra_ops=reward_ops)

    @staticmethod
    def _apply_ops(store, ops) -> None:
        if ops:
            with store._tx():
                for sql, params in ops:
                    store.conn.execute(sql, params)

    def _prior(self, store, model: str, node_type: str) -> float:
        from argus.graph import registry

        v = registry.profile(store.conn, model, node_type, "quality")
        if v is None:
            v = registry.profile(store.conn, model, "*", "quality")  # same fallback as the router
        return float(v if v is not None else self.cfg.get("defaults", {}).get("quality", 0.5))

    def _reward_ops(self, store, node, r) -> list:
        """Rewards for this node's result, as ops. Arms are always seeded from
        the profiler prior, in every mode, so fixed-mode evidence warms the
        same posterior learning mode reads (review M6). Baseline records none:
        it is the no-learning control."""
        if self.mode == "baseline":
            return []
        strength = float(self.cfg.get("prior_strength", 6))
        ops = []
        for node_type, model, source, reward, detail, suffix in r.rewards:
            ops += bandit.reward_ops(node_type=node_type, model=model, reward_key=node.node_id + suffix,
                                     source=source, reward=reward, detail=detail,
                                     prior_quality=self._prior(store, model, node_type), strength=strength)
        if node.node_type == "critique" and r.output and r.output.get("verdict"):
            # Critic reward goes to the model that AUTHORED the judged output,
            # and only when a different model judged it: a model never grades
            # itself.
            critic_model = r.output.get("_produced_by")
            for e in store.edges(node.graph_id):
                if e.dst == node.node_id and e.channel == "subject":
                    subj = store.node(e.src)
                    author = (subj.output or {}).get("_produced_by")
                    if author and critic_model and author != critic_model:
                        ops += bandit.reward_ops(
                            node_type=subj.node_type, model=author, reward_key=e.src, source="critic",
                            reward=1.0 if r.output["verdict"].get("verdict") == "pass" else 0.0,
                            detail=f"judged by {critic_model}",
                            prior_quality=self._prior(store, author, subj.node_type), strength=strength)
        return ops

    def _apply_plan(self, store, g, target, node, output, extra_ops=()) -> None:
        plan = output.get("plan") or {}
        nodes, edges = store.nodes(g.graph_id), store.edges(g.graph_id)
        spec = GraphSpec()
        changes_from: List[str] = []
        dropped: List[str] = []
        for step in plan.get("steps", []):
            name, params = step.get("template"), dict(step.get("params") or {})
            if name not in templates.TEMPLATES or not templates.TEMPLATES[name].planner_visible:
                dropped.append(f"unknown template {name!r}")
                continue
            if name == "check_page" and not handlers.same_site(params.get("url") or "", target.get("url", "")):
                dropped.append(f"off-site url {params.get('url')!r}")  # injection defence
                continue
            if name == "follow_link":
                trig = (store.node(node.node_id).params or {})
                params.setdefault("fields", trig.get("fields") or target.get("extract") or [])
                if params.get("from_node") not in nodes:
                    dropped.append("follow_link from an unknown node")
                    continue
            params["fields"] = [f for f in params.get("fields") or [] if f in (target.get("extract") or params["fields"])]
            if templates.TEMPLATES[name].missing_params(params):
                dropped.append(f"{name}: missing {templates.TEMPLATES[name].missing_params(params)}")
                continue
            sub, ex = templates.instantiate(name, g.graph_id, target["id"], params)
            spec = spec.merged(sub)
            if "changes" in ex:
                changes_from.append(ex["changes"])
        summarize = [n for n in nodes.values() if n.node_type == "summarize"]
        if node.node_type == "plan" and not summarize:
            if not changes_from:
                fallback, ex = templates.instantiate("check_page", g.graph_id, target["id"], {
                    "url": target.get("url"), "fields": list(target.get("extract") or [])})
                spec = spec.merged(fallback)
                changes_from.append(ex["changes"])
                dropped.append("no usable steps; fell back to the target page")
            dig, dex = templates.instantiate("digest", g.graph_id, target["id"], {"changes_from": changes_from})
            spec = spec.merged(dig)
            if target.get("notify"):
                notify, _ = templates.instantiate("notify", g.graph_id, target["id"], {"digest_from": dex["digest"]})
                spec = spec.merged(notify)
        elif summarize and changes_from and summarize[0].status == "pending":
            for c in changes_from:
                spec.edges.append(EdgeSpec(c, summarize[0].node_id, "changes"))
        counts = node.node_type == "replan"
        g_now = store.graph(g.graph_id)
        violations = validator.validate_expansion(
            store.spec(g.graph_id), spec, g_now.limits, g_now.expansions_used + (1 if counts else 0))
        if violations and node.node_type == "plan":
            # Swap in the deterministic fallback BEFORE writing, so the plan's
            # output and the graph it produced land in ONE transaction (review
            # M4: two transactions could leave a done plan with nothing to run).
            dropped.append("planned graph rejected: " + "; ".join(str(v) for v in violations[:5]))
            fb, ex = templates.instantiate("check_page", g.graph_id, target["id"], {
                "url": target.get("url"), "fields": list(target.get("extract") or [])})
            dig, dex = templates.instantiate("digest", g.graph_id, target["id"], {"changes_from": [ex["changes"]]})
            spec = fb.merged(dig)
            if target.get("notify"):
                spec = spec.merged(templates.instantiate("notify", g.graph_id, target["id"],
                                                         {"digest_from": dex["digest"]})[0])
        out = {**output, "dropped_steps": dropped}
        store.apply_expansion(
            g.graph_id, spec, kind=node.node_type, requested_by=node.node_id,
            expansion_id=f"exp-{node.node_id}", counts_toward_limit=counts,
            complete_node=(node.node_id, out), extra_ops=extra_ops)

    def _record_templates(self, store, graph_id) -> None:
        from argus.graph import procedural

        procedural.record_graph(store, graph_id)

    def _outcome(self, store, g, status) -> RunOutcome:
        nodes = store.nodes(g.graph_id)
        counts: Dict[str, int] = {}
        for n in nodes.values():
            counts[n.status] = counts.get(n.status, 0) + 1
        digest = None
        for n in nodes.values():
            if n.node_type == "summarize" and n.status == "done":
                digest = (n.output or {}).get("digest", {}).get("digest")
        return RunOutcome(g.run_id, g.graph_id, status, digest, counts)
