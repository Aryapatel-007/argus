"""Graph validator: unit tests plus hypothesis property tests.

The property tests check the validator against an INDEPENDENT oracle (brute
force enumeration of every root-to-node path), not against itself. The
negative control at the bottom proves the gate-bypass property can fail:
with the gate check removed, hypothesis must find a graph that sneaks an
execute_action past the gate and still validates.
"""

from __future__ import annotations

from typing import Dict, List, Set

import pytest
from hypothesis import HealthCheck, find, given, settings
from hypothesis import strategies as st

from argus.graph import nodes as node_types
from argus.graph import templates, validator
from argus.graph.model import EdgeSpec, GraphSpec, Limits, NodeSpec


# ---------------------------------------------------------------------------
# independent oracle
# ---------------------------------------------------------------------------


def oracle_has_gate_free_path(graph: GraphSpec) -> bool:
    """True if some root -> execute_action path contains no approval_gate.

    Enumerates paths by DFS. Exponential in general, fine at test sizes.
    Deliberately shares no code with validator.gate_bypasses.
    """
    children: Dict[str, List[str]] = {n: [] for n in graph.nodes}
    for e in graph.edges:
        if e.src in children and e.dst in graph.nodes:
            children[e.src].append(e.dst)

    def walk(node: str, seen: Set[str]) -> bool:
        if graph.nodes[node].node_type == "approval_gate":
            return False
        if graph.nodes[node].node_type == "execute_action":
            return True
        return any(walk(c, seen | {c}) for c in children[node] if c not in seen)

    return any(walk(r, {r}) for r in graph.roots())


def oracle_has_cycle(graph: GraphSpec) -> bool:
    color: Dict[str, int] = {n: 0 for n in graph.nodes}
    adj: Dict[str, List[str]] = {n: [] for n in graph.nodes}
    for e in graph.edges:
        adj[e.src].append(e.dst)

    def dfs(n: str) -> bool:
        color[n] = 1
        for m in adj[n]:
            if color[m] == 1 or (color[m] == 0 and dfs(m)):
                return True
        color[n] = 2
        return False

    return any(color[n] == 0 and dfs(n) for n in graph.nodes)


# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------

PALETTE = ["fetch_page", "extract", "critique", "summarize", "propose_action",
           "approval_gate", "execute_action", "diff", "recall"]


def _channels(src_type: str, dst_type: str) -> List[str]:
    src = node_types.get(src_type)
    return [c for c in src.provides if node_types.accepts(dst_type, c)]


@st.composite
def dags(draw, max_nodes: int = 9, allow_any_channel: bool = True):
    n = draw(st.integers(min_value=1, max_value=max_nodes))
    types = [draw(st.sampled_from(PALETTE)) for _ in range(n)]
    nodes = []
    for i, t in enumerate(types):
        key = f"k{i}" if node_types.get(t).side_effect else None
        nodes.append(NodeSpec(node_id=f"n{i}", node_type=t, idempotency_key=key))
    edges = []
    for j in range(n):
        for i in range(j):
            if draw(st.booleans()):
                valid = _channels(types[i], types[j])
                if valid:
                    channel = draw(st.sampled_from(valid))
                elif allow_any_channel:
                    channel = draw(st.sampled_from(["page", "approval", "proposal", "digest"]))
                else:
                    continue
                edges.append(EdgeSpec(f"n{i}", f"n{j}", channel))
    return GraphSpec.build(nodes, edges)


def codes(violations) -> Set[str]:
    return {v.code for v in violations}


# ---------------------------------------------------------------------------
# properties
# ---------------------------------------------------------------------------


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(dags())
def test_gate_bypass_reported_iff_oracle_finds_gate_free_path(graph):
    reported = "gate_bypass" in codes(validator.validate(graph))
    assert reported == oracle_has_gate_free_path(graph)


@settings(max_examples=300, deadline=None)
@given(dags())
def test_a_graph_that_validates_never_lets_an_action_skip_the_gate(graph):
    if not validator.validate(graph):
        assert not oracle_has_gate_free_path(graph)


@settings(max_examples=300, deadline=None)
@given(dags(), st.data())
def test_any_back_edge_is_rejected_as_a_cycle(graph, data):
    ids = sorted(graph.nodes, key=lambda x: int(x[1:]))
    if len(ids) < 2:
        return
    j = data.draw(st.integers(min_value=1, max_value=len(ids) - 1))
    i = data.draw(st.integers(min_value=0, max_value=j))
    looped = graph.copy()
    looped.edges.append(EdgeSpec(ids[j], ids[i], "page"))
    if oracle_has_cycle(looped):
        assert "cycle" in codes(validator.validate(looped))


@settings(max_examples=300, deadline=None)
@given(dags())
def test_forward_only_graphs_are_never_called_cyclic(graph):
    assert "cycle" not in codes(validator.validate(graph))


@settings(max_examples=300, deadline=None)
@given(dags(), dags(max_nodes=5))
def test_expansion_cannot_introduce_a_bypass(existing, extra):
    # Rename the expansion's nodes so ids are fresh, and let it attach to an
    # arbitrary existing node (the way a replan would).
    renamed = GraphSpec.build(
        [n.clone(node_id="x" + n.node_id, retry_root="x" + n.node_id) for n in extra.nodes.values()],
        [EdgeSpec("x" + e.src, "x" + e.dst, e.channel) for e in extra.edges],
    )
    anchor = sorted(existing.nodes)[0]
    first_new = sorted(renamed.nodes)[0]
    renamed.edges.append(EdgeSpec(anchor, first_new, "page"))
    violations = validator.validate_expansion(existing, renamed)
    merged = existing.merged(renamed)
    if not violations:
        assert not oracle_has_gate_free_path(merged)


# ---------------------------------------------------------------------------
# unit tests on hand-built graphs
# ---------------------------------------------------------------------------


def _g(spec, edges):
    nodes = [NodeSpec(node_id=i, node_type=t,
                      idempotency_key=(f"key-{i}" if node_types.get(t).side_effect else None))
             for i, t in spec]
    return GraphSpec.build(nodes, [EdgeSpec(*e) for e in edges])


def test_direct_bypass_is_rejected():
    g = _g([("p", "propose_action"), ("x", "execute_action")], [("p", "x", "approval")])
    assert "gate_bypass" in codes(validator.validate(g))


def test_bypass_through_a_second_parent_is_rejected():
    # Gate on one path is not enough: every path must cross a gate.
    g = _g([("s", "summarize"), ("p", "propose_action"), ("g", "approval_gate"),
            ("x", "execute_action"), ("d", "diff")],
           [("d", "s", "changes"), ("s", "p", "digest"), ("p", "g", "proposal"),
            ("g", "x", "approval"), ("d", "x", "approval")])
    assert "gate_bypass" in codes(validator.validate(g))


def test_orphan_action_is_rejected():
    g = _g([("x", "execute_action")], [])
    v = codes(validator.validate(g))
    assert "gate_bypass" in v and "unsatisfied_input" in v


def test_side_effect_without_key_is_rejected():
    g = GraphSpec.build([NodeSpec("g", "approval_gate")], [])
    assert "missing_idempotency_key" in codes(validator.validate(g))


def test_prunable_gate_is_rejected():
    g = GraphSpec.build([NodeSpec("g", "approval_gate", idempotency_key="k", prunable=True)], [])
    assert "prunable_side_effect" in codes(validator.validate(g))


def test_duplicate_keys_are_rejected():
    g = _g([("g", "approval_gate"), ("h", "approval_gate")], [])
    g.nodes["h"].idempotency_key = g.nodes["g"].idempotency_key
    assert "duplicate_idempotency_key" in codes(validator.validate(g))


def test_limits_are_enforced():
    nodes = [NodeSpec(f"r{i}", "recall") for i in range(6)]
    g = GraphSpec.build(nodes, [])
    assert "node_limit" in codes(validator.validate(g, Limits(max_nodes=5)))
    assert "expansion_limit" in codes(validator.validate(g, Limits(max_expansions=2), expansions_used=3))
    deep = _g([("a", "fetch_page"), ("b", "extract"), ("c", "critique")],
              [("a", "b", "page"), ("b", "c", "subject"), ("a", "c", "evidence")])
    assert "depth_limit" in codes(validator.validate(deep, Limits(max_depth=1)))
    deep.nodes["b"].attempt = 9
    assert "attempt_limit" in codes(validator.validate(deep))


def test_channel_mismatch_is_rejected():
    g = _g([("r", "recall"), ("e", "extract")], [("r", "e", "page")])
    assert "channel_not_provided" in codes(validator.validate(g))


def test_ambiguous_single_input_is_rejected():
    g = _g([("a", "fetch_page"), ("b", "fetch_page"), ("e", "extract")],
           [("a", "e", "page"), ("b", "e", "page")])
    assert "ambiguous_input" in codes(validator.validate(g))


def test_retry_chain_feeding_one_input_is_fine():
    g = _g([("a", "fetch_page"), ("e", "extract")], [("a", "e", "page")])
    g.nodes["a2"] = NodeSpec("a2", "fetch_page", attempt=2, retry_of="a", retry_root="a")
    g.edges.append(EdgeSpec("a2", "e", "page"))
    assert validator.validate(g) == []


@pytest.mark.parametrize("name", ["check_page", "notify", "digest", "reflect"])
def test_every_builtin_template_validates_in_context(name):
    page, exports = templates.instantiate("check_page", "g1", "t1",
                                          {"url": "http://x/", "fields": ["fees"]})
    graph = page
    if name == "digest":
        sub, _ = templates.instantiate("digest", "g1", "t1", {"changes_from": [exports["changes"]]})
        graph = graph.merged(sub)
    elif name == "notify":
        dig, dex = templates.instantiate("digest", "g1", "t1", {"changes_from": [exports["changes"]]})
        sub, _ = templates.instantiate("notify", "g1", "t1", {"digest_from": dex["digest"]})
        graph = graph.merged(dig).merged(sub)
    elif name == "reflect":
        graph, _ = templates.instantiate("reflect", "g1", "t1", {})
    assert validator.validate(graph) == []


def test_planner_expansion_may_not_rewire_an_existing_gate():
    dig_page, ex = templates.instantiate("check_page", "g1", "t1", {"url": "http://x/", "fields": ["a"]})
    dig, dex = templates.instantiate("digest", "g1", "t1", {"changes_from": [ex["changes"]]})
    notify, nex = templates.instantiate("notify", "g1", "t1", {"digest_from": dex["digest"]})
    existing = dig_page.merged(dig).merged(notify)
    sneaky = GraphSpec.build([NodeSpec("p2", "propose_action")],
                             [EdgeSpec(dex["digest"], "p2", "digest"),
                              EdgeSpec("p2", nex["action"], "approval")])
    v = codes(validator.validate_expansion(existing, sneaky))
    assert "rewires_existing" in v


def test_open_fan_in_may_receive_a_new_page():
    page, ex = templates.instantiate("check_page", "g1", "t1", {"url": "http://x/", "fields": ["a"]})
    dig, dex = templates.instantiate("digest", "g1", "t1", {"changes_from": [ex["changes"]]})
    existing = page.merged(dig)
    page2, ex2 = templates.instantiate("check_page", "g1", "t1", {"url": "http://y/", "fields": ["a"]})
    page2.edges.append(EdgeSpec(ex2["changes"], dex["digest"], "changes"))
    assert validator.validate_expansion(existing, page2) == []
    existing.nodes[dex["digest"]].status = "running"
    assert "rewires_existing" in codes(validator.validate_expansion(existing, page2))


# ---------------------------------------------------------------------------
# NEGATIVE CONTROL
# ---------------------------------------------------------------------------


NEG_SETTINGS = settings(max_examples=5000, deadline=None,
                        suppress_health_check=list(HealthCheck))


def test_negative_control_iff_property_fails_without_the_gate_check(monkeypatch):
    """With gate_bypasses disabled, the iff property must have a counterexample.

    If hypothesis could not find one, the iff property above would be passing
    for a reason other than the gate check, i.e. it would be vacuous.
    """
    monkeypatch.setattr(validator, "gate_bypasses", lambda graph: [])
    found = find(
        dags(),
        lambda g: ("gate_bypass" in codes(validator.validate(g))) != oracle_has_gate_free_path(g),
        settings=NEG_SETTINGS,
    )
    assert oracle_has_gate_free_path(found)


def test_negative_control_clean_graph_can_bypass_when_both_defences_are_off(monkeypatch):
    """The gate is defended twice: the structural path check, and the fact
    that only approval_gate provides the `approval` channel. Turn BOTH off and
    a graph with a gate-free path to execute_action must validate clean.
    This is what the 'validates => no bypass' property protects against."""
    monkeypatch.setattr(validator, "gate_bypasses", lambda graph: [])
    real_validate = validator.validate
    ignored = {"channel_not_provided", "channel_not_accepted"}

    def broken(graph, *a, **kw):
        return [x for x in real_validate(graph, *a, **kw) if x.code not in ignored]

    found = find(
        dags(),
        lambda g: oracle_has_gate_free_path(g) and not broken(g),
        settings=NEG_SETTINGS,
    )
    assert oracle_has_gate_free_path(found)
    # And with the real validator the same graph is rejected.
    monkeypatch.undo()
    assert validator.validate(found)
