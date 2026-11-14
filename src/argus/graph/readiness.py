"""Readiness, input resolution, skip cascades and retry planning. Pure.

Everything here is a function of the stored graph (nodes + edges), so the
engine can recompute it after a crash and get the same answer. Nothing here
writes; the engine hands the results to `store`.

Retry chains (D-013). A retried node X gets a clone X' with attempt+1 and
retry_root = X's root. All members of a chain feed the same children. For
one input channel of a node:
  - satisfied: some chain member is done and no member with a HIGHER attempt
    is still open. The value is that done member's output.
  - open: otherwise, while any member is not terminal (waiting).
  - dead: every member terminal and none done (unsatisfiable -> skip).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from argus.graph import nodes as node_types
from argus.graph.model import TERMINAL_NODE_STATUSES, EdgeSpec, GraphSpec, Limits, NodeSpec

OPEN = frozenset({"pending", "ready", "running", "blocked"})
NO_RETRY_TYPES = frozenset({"approval_gate", "execute_action", "plan"})


@dataclass
class Sweep:
    ready: List[str] = field(default_factory=list)
    skip: Dict[str, str] = field(default_factory=dict)


def _parents_by_channel(node_id: str, edges: List[EdgeSpec]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    for e in edges:
        if e.dst == node_id:
            out[e.channel].append(e.src)
    return out


def _chains(sources: List[str], nodes: Dict[str, Any]) -> Dict[str, List[Any]]:
    chains: Dict[str, List[Any]] = defaultdict(list)
    for s in sources:
        if s in nodes:
            chains[nodes[s].retry_root].append(nodes[s])
    return chains


def chain_state(members: List[Any]) -> Tuple[str, Optional[Any]]:
    done = [m for m in members if m.status == "done"]
    best = max(done, key=lambda m: m.attempt) if done else None
    if best is not None:
        if any(m.status in OPEN and m.attempt > best.attempt for m in members):
            return "open", None
        return "satisfied", best
    if any(m.status in OPEN for m in members):
        return "open", None
    return "dead", None


def input_states(node_id: str, nodes: Dict[str, Any], edges: List[EdgeSpec]
                 ) -> Dict[str, Tuple[str, Any]]:
    """channel -> (state, value). value is a payload (single) or list (fan-in)."""
    nt = node_types.get(nodes[node_id].node_type)
    by_channel = _parents_by_channel(node_id, edges)
    result: Dict[str, Tuple[str, Any]] = {}
    for channel in nt.inputs:
        chains = _chains(by_channel.get(channel, []), nodes)
        if not chains:
            result[channel] = ("satisfied", None) if channel in nt.optional_inputs else ("dead", None)
            continue
        # The validator guarantees one chain for a non-fan-in input.
        state, best = chain_state(next(iter(chains.values())))
        result[channel] = (state, (best.output or {}).get(channel) if best else None)
    for channel in nt.fan_in:
        chains = _chains(by_channel.get(channel, []), nodes)
        values, any_open = [], False
        for root in sorted(chains):
            state, best = chain_state(chains[root])
            if state == "open":
                any_open = True
            elif state == "satisfied":
                value = (best.output or {}).get(channel)
                if value is not None:
                    values.append(value)
        result[channel] = ("open", None) if any_open else ("satisfied", values)
    return result


def resolve_inputs(node_id: str, nodes: Dict[str, Any], edges: List[EdgeSpec]) -> Dict[str, Any]:
    return {c: v for c, (s, v) in input_states(node_id, nodes, edges).items() if s == "satisfied"}


def gate_ancestors(node_id: str, nodes: Dict[str, Any], edges: List[EdgeSpec]) -> List[str]:
    parents: Dict[str, List[str]] = defaultdict(list)
    for e in edges:
        parents[e.dst].append(e.src)
    seen: Set[str] = set()
    stack = list(parents[node_id])
    gates = []
    while stack:
        n = stack.pop()
        if n in seen or n not in nodes:
            continue
        seen.add(n)
        if nodes[n].node_type == "approval_gate":
            gates.append(n)
        stack.extend(parents[n])
    return sorted(gates)


def sweep(nodes: Dict[str, Any], edges: List[EdgeSpec]) -> Sweep:
    """Which pending nodes become ready, which can never run (skip)."""
    out = Sweep()
    planning = any(
        n.node_type in ("plan", "replan") and n.status not in TERMINAL_NODE_STATUSES
        for n in nodes.values()
    )
    for node_id in sorted(nodes):
        node = nodes[node_id]
        if node.status != "pending":
            continue
        nt = node_types.get(node.node_type)
        states = input_states(node_id, nodes, edges)
        dead = [c for c, (s, _) in states.items() if s == "dead"]
        if dead:
            out.skip[node_id] = f"upstream input {dead[0]!r} can never be satisfied"
            continue
        if any(s == "open" for s, _ in states.values()):
            continue
        if nt.fan_in and planning:
            continue  # a running planner may still add parents to this fan-in
        if node.node_type == "execute_action":
            gates = gate_ancestors(node_id, nodes, edges)
            # Structural gate (ARGUS_BUILD 6.5): every gate ancestor approved.
            # A skipped (rejected) gate kills the action via the dead input above.
            if not gates or any(nodes[g].status != "done" for g in gates):
                continue
        out.ready.append(node_id)
    return out


# ---------------------------------------------------------------------------
# retries
# ---------------------------------------------------------------------------


def _reflection_from(node: Any) -> str:
    out = node.output or {}
    verdict = out.get("verdict") if isinstance(out, dict) else None
    if isinstance(verdict, dict) and verdict.get("reflection"):
        return str(verdict["reflection"])
    return (node.error or "previous attempt failed")[:800]


def retry_id(node: Any) -> str:
    return f"{node.retry_root}~a{node.attempt + 1}"


def plan_retries(nodes: Dict[str, Any], edges: List[EdgeSpec], limits: Limits
                 ) -> List[Tuple[str, GraphSpec]]:
    """Expansions that retry failed nodes. Deterministic ids: the same stored
    state always yields the same expansion, so a crash between a failure and
    its retry being stored is repaired on resume by re-planning it."""
    retried = {n.retry_of for n in nodes.values() if n.retry_of}
    out: List[Tuple[str, GraphSpec]] = []
    for node_id in sorted(nodes):
        f = nodes[node_id]
        if f.status != "failed" or node_id in retried or f.node_type in NO_RETRY_TYPES:
            continue
        if f.attempt >= limits.max_attempts:
            continue
        reflection = _reflection_from(f)
        sub_nodes: Dict[str, NodeSpec] = {}
        sub_edges: List[EdgeSpec] = []

        def clone(src: Any, extra_params: Dict[str, Any]) -> NodeSpec:
            spec = NodeSpec(
                node_id=retry_id(src), node_type=src.node_type,
                params={**src.params, **extra_params}, attempt=src.attempt + 1,
                retry_of=src.node_id, retry_root=src.retry_root,
                prunable=src.prunable, difficulty=src.difficulty,
                template_instance=src.template_instance,
            )
            sub_nodes[spec.node_id] = spec
            return spec

        if f.node_type == "critique":
            subj_sources = [e.src for e in edges if e.dst == node_id and e.channel == "subject"]
            subject = max((nodes[s] for s in subj_sources if nodes[s].status == "done"),
                          key=lambda n: n.attempt, default=None)
            if subject is None or subject.attempt >= limits.max_attempts:
                continue
            s2 = clone(subject, {"reflection": reflection})
            c2 = clone(f, {})
            for e in edges:
                if e.dst == subject.node_id:
                    sub_edges.append(EdgeSpec(e.src, s2.node_id, e.channel))
                if e.src == subject.node_id and e.dst != node_id:
                    sub_edges.append(EdgeSpec(s2.node_id, e.dst, e.channel))
                if e.dst == node_id and e.channel != "subject":
                    sub_edges.append(EdgeSpec(e.src, c2.node_id, e.channel))
                if e.src == node_id:
                    sub_edges.append(EdgeSpec(c2.node_id, e.dst, e.channel))
            sub_edges.append(EdgeSpec(s2.node_id, c2.node_id, "subject"))
        else:
            f2 = clone(f, {"reflection": reflection})
            for e in edges:
                if e.dst == node_id:
                    sub_edges.append(EdgeSpec(e.src, f2.node_id, e.channel))
                if e.src == node_id:
                    sub_edges.append(EdgeSpec(f2.node_id, e.dst, e.channel))
        out.append((f"retry-{node_id}", GraphSpec(sub_nodes, sub_edges)))
    return out


def graph_outcome(nodes: Dict[str, Any]) -> Optional[str]:
    """None while work remains; else 'waiting_approval', 'failed' or 'done'."""
    statuses = {n.status for n in nodes.values()}
    if statuses & {"ready", "running"}:
        return None
    if "blocked" in statuses:
        # Pending nodes behind a blocked gate wait for a human, not for work.
        return "waiting_approval"
    if "pending" in statuses:
        return None
    # A chain that ended with no done member, on a node that mattered.
    chains: Dict[str, List[Any]] = defaultdict(list)
    for n in nodes.values():
        chains[n.retry_root].append(n)
    for members in chains.values():
        if any(m.status == "done" for m in members):
            continue
        if any(m.status == "failed" for m in members) and not any(m.prunable for m in members):
            return "failed"
    return "done"
