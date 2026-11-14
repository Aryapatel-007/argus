"""Structural validation. If the graph fails, nothing runs.

Pure functions over `GraphSpec`. Called on the initial graph and on
(existing graph + proposed expansion) before an expansion is stored, so a
planner can never sneak in a structure the engine would refuse at start.

The approval gate is enforced HERE, by structure, not by trusting the
planner's labels: every `execute_action` must be unreachable from any root
once approval gates are deleted from the graph. The scheduler enforces the
second half (the gate must be approved before the action is ready) and the
action executor checks the approvals row a third time at run time.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from argus.graph import nodes as node_types
from argus.graph.model import GraphSpec, Limits

GATE = "approval_gate"
ACTION = "execute_action"


@dataclass(frozen=True)
class Violation:
    code: str
    message: str
    node_id: Optional[str] = None

    def __str__(self) -> str:
        where = f" [{self.node_id}]" if self.node_id else ""
        return f"{self.code}{where}: {self.message}"


class GraphInvalid(ValueError):
    def __init__(self, violations: List[Violation]):
        self.violations = violations
        super().__init__("; ".join(str(v) for v in violations[:10]))


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------


def topological_order(graph: GraphSpec) -> Optional[List[str]]:
    """Kahn's algorithm. None when the graph has a cycle."""
    indegree: Dict[str, int] = {n: 0 for n in graph.nodes}
    out: Dict[str, List[str]] = defaultdict(list)
    for e in graph.edges:
        if e.src in indegree and e.dst in indegree:
            indegree[e.dst] += 1
            out[e.src].append(e.dst)
    queue = deque(sorted(n for n, d in indegree.items() if d == 0))
    order: List[str] = []
    while queue:
        n = queue.popleft()
        order.append(n)
        for m in out[n]:
            indegree[m] -= 1
            if indegree[m] == 0:
                queue.append(m)
    return order if len(order) == len(graph.nodes) else None


def depths(graph: GraphSpec, order: List[str]) -> Dict[str, int]:
    """Longest path from any root, in edges."""
    depth = {n: 0 for n in graph.nodes}
    out: Dict[str, List[str]] = defaultdict(list)
    for e in graph.edges:
        out[e.src].append(e.dst)
    for n in order:
        for m in out[n]:
            depth[m] = max(depth[m], depth[n] + 1)
    return depth


def gate_bypasses(graph: GraphSpec) -> List[str]:
    """execute_action nodes reachable from a root without passing a gate.

    Also returns an action with no gate ancestor at all (it is its own root,
    or every path to it avoids gates). Reachability starts at every root of
    the FULL graph and never steps onto a gate node.
    """
    out: Dict[str, List[str]] = defaultdict(list)
    for e in graph.edges:
        out[e.src].append(e.dst)

    def type_of(n: str) -> str:
        return graph.nodes[n].node_type

    reached: Set[str] = set()
    queue = deque(r for r in graph.roots() if type_of(r) != GATE)
    while queue:
        n = queue.popleft()
        if n in reached:
            continue
        reached.add(n)
        for m in out[n]:
            if m in graph.nodes and type_of(m) != GATE and m not in reached:
                queue.append(m)
    return sorted(n for n in reached if type_of(n) == ACTION)


# ---------------------------------------------------------------------------
# the validator
# ---------------------------------------------------------------------------


def validate(graph: GraphSpec, limits: Limits = Limits(), expansions_used: int = 0) -> List[Violation]:
    v: List[Violation] = []

    # -- node types and per-node fields ------------------------------------
    for node_id, node in graph.nodes.items():
        if node.node_type not in node_types.NODE_TYPES:
            v.append(Violation("unknown_type", f"unknown node type {node.node_type!r}", node_id))
            continue
        nt = node_types.get(node.node_type)
        if nt.side_effect:
            if not node.idempotency_key:
                v.append(Violation("missing_idempotency_key",
                                   "side-effect node has no idempotency key", node_id))
            if node.prunable:
                v.append(Violation("prunable_side_effect",
                                   "gates and side-effect nodes can never be prunable", node_id))
        if node.attempt < 1 or node.attempt > limits.max_attempts:
            v.append(Violation("attempt_limit",
                               f"attempt {node.attempt} outside 1..{limits.max_attempts}", node_id))
    if v and any(x.code == "unknown_type" for x in v):
        return v  # later checks assume known types

    keys = [n.idempotency_key for n in graph.nodes.values() if n.idempotency_key]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    for k in dupes:
        v.append(Violation("duplicate_idempotency_key", f"key {k} used more than once"))

    # -- edges -------------------------------------------------------------
    for e in graph.edges:
        if e.src not in graph.nodes or e.dst not in graph.nodes:
            v.append(Violation("dangling_edge", f"edge {e.src}->{e.dst} names a missing node"))
            continue
        if e.src == e.dst:
            v.append(Violation("self_loop", "edge from a node to itself", e.src))
        src_t = node_types.get(graph.nodes[e.src].node_type)
        if e.channel not in src_t.provides:
            v.append(Violation("channel_not_provided",
                               f"{src_t.name} does not provide {e.channel!r}", e.src))
        if not node_types.accepts(graph.nodes[e.dst].node_type, e.channel):
            v.append(Violation("channel_not_accepted",
                               f"{graph.nodes[e.dst].node_type} does not accept {e.channel!r}", e.dst))

    # -- input satisfiability ----------------------------------------------
    by_dst: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    for e in graph.edges:
        if e.src in graph.nodes:
            by_dst[e.dst][e.channel].add(e.src)
    for node_id, node in graph.nodes.items():
        nt = node_types.get(node.node_type)
        for channel in nt.inputs + nt.fan_in:
            sources = by_dst[node_id].get(channel, set())
            if not sources and channel not in nt.optional_inputs:
                v.append(Violation("unsatisfied_input", f"input {channel!r} has no parent", node_id))
            if channel in nt.inputs and sources:
                chains = {graph.nodes[s].retry_root for s in sources}
                if len(chains) > 1:
                    v.append(Violation("ambiguous_input",
                                       f"input {channel!r} fed by {len(chains)} independent nodes", node_id))

    # -- structure ---------------------------------------------------------
    order = topological_order(graph)
    if order is None:
        v.append(Violation("cycle", "graph is not a DAG; retries must be new nodes, not back edges"))
    else:
        d = depths(graph, order)
        too_deep = [n for n, x in d.items() if x > limits.max_depth]
        for n in too_deep:
            v.append(Violation("depth_limit", f"depth {d[n]} > {limits.max_depth}", n))

    for n in gate_bypasses(graph):
        v.append(Violation("gate_bypass",
                           "execute_action reachable without passing an approval_gate", n))

    if len(graph.nodes) > limits.max_nodes:
        v.append(Violation("node_limit", f"{len(graph.nodes)} nodes > {limits.max_nodes}"))
    if expansions_used > limits.max_expansions:
        v.append(Violation("expansion_limit", f"{expansions_used} expansions > {limits.max_expansions}"))

    return v


def check(graph: GraphSpec, limits: Limits = Limits(), expansions_used: int = 0) -> None:
    violations = validate(graph, limits, expansions_used)
    if violations:
        raise GraphInvalid(violations)


def validate_expansion(
    existing: GraphSpec,
    expansion: GraphSpec,
    limits: Limits = Limits(),
    expansions_used: int = 0,
    allow_existing_dst: bool = False,
) -> List[Violation]:
    """Validate the graph as it WOULD be after the expansion.

    Planner expansions may only add edges INTO new nodes, never rewire the
    inputs of a node that already exists. One exception: a FAN-IN channel of
    an existing node that has not started yet (status pending), e.g. a new
    page's diff feeding the run's summarize. That cannot weaken a gate: a
    gate's and an action's inputs are not fan-in. `allow_existing_dst` is for
    the engine's own retry wiring only. Node ids must be new.
    """
    v: List[Violation] = []
    clash = sorted(set(expansion.nodes) & set(existing.nodes))
    for n in clash:
        v.append(Violation("node_id_reused", "expansion reuses an existing node id", n))
    if not allow_existing_dst:
        for e in expansion.edges:
            if e.dst not in existing.nodes:
                continue
            dst = existing.nodes[e.dst]
            open_fan_in = (
                dst.status == "pending"
                and dst.node_type in node_types.NODE_TYPES
                and e.channel in node_types.get(dst.node_type).fan_in
            )
            if not open_fan_in:
                v.append(Violation("rewires_existing",
                                   f"expansion adds input {e.channel!r} to existing node", e.dst))
    if v:
        return v
    return validate(existing.merged(expansion), limits, expansions_used)
