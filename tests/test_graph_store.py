"""Graph store (persistence) and readiness (pure) tests."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest
from hypothesis import HealthCheck, find, given, settings
from hypothesis import strategies as st

from argus.graph import readiness, templates
from argus.graph.model import EdgeSpec, GraphSpec, Limits, NodeSpec
from argus.graph.store import GraphStore, IllegalTransition
from argus.storage import db
from argus.storage.migrate import apply_migrations


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "g.db"
    apply_migrations(path)
    with db.connect(path) as conn:
        with db.atomic(conn):
            conn.execute("INSERT INTO runs (run_id, target_id, status) VALUES ('r1', 't1', 'running')")
        yield GraphStore(conn)


def _page_graph(store, fields=("fees",)):
    gid = store.create_graph(run_id="r1", target_id="t1", kind="watch", limits=Limits())
    sub, ex = templates.instantiate("check_page", gid, "t1", {"url": "http://127.0.0.1/x", "fields": list(fields)})
    ok, v = store.apply_expansion(gid, sub, kind="initial", counts_toward_limit=False)
    assert ok, v
    return gid, sub, ex


def _by_type(nodes, t):
    return [n for n in nodes.values() if n.node_type == t]


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


def test_expansion_stores_nodes_edges_and_creation_events(store):
    gid, sub, _ = _page_graph(store)
    nodes = store.nodes(gid)
    assert set(nodes) == set(sub.nodes)
    assert all(n.status == "pending" for n in nodes.values())
    assert len(store.edges(gid)) == len(sub.edges)
    for n in nodes:
        events = store.events(n)
        assert [e["seq"] for e in events] == [1] and events[0]["to_status"] == "pending"


def test_rejected_expansion_is_recorded_and_writes_no_nodes(store):
    gid, _, _ = _page_graph(store)
    bad = GraphSpec.build([NodeSpec("x1", "execute_action", idempotency_key="k")], [])
    ok, violations = store.apply_expansion(gid, bad, kind="replan")
    assert not ok and any(v.code == "gate_bypass" for v in violations)
    assert "x1" not in store.nodes(gid)
    row = store.conn.execute("SELECT accepted FROM graph_expansions WHERE kind='replan'").fetchone()
    assert row["accepted"] == 0


def test_reapplying_the_same_expansion_id_is_a_no_op(store):
    gid, _, _ = _page_graph(store)
    extra, _ = templates.instantiate("check_page", gid, "t1", {"url": "http://127.0.0.1/y", "fields": ["a"]})
    assert store.apply_expansion(gid, extra, kind="replan", expansion_id="e1")[0]
    before = len(store.nodes(gid))
    assert store.apply_expansion(gid, extra, kind="replan", expansion_id="e1")[0]
    assert len(store.nodes(gid)) == before
    assert store.graph(gid).expansions_used == 1


def test_transition_writes_status_output_and_event_together(store):
    gid, _, _ = _page_graph(store)
    fetch = _by_type(store.nodes(gid), "fetch_page")[0]
    store.transition(fetch.node_id, "ready")
    store.start_node(fetch.node_id)
    store.transition(fetch.node_id, "done", output={"page": {"text": "x"}})
    row = store.node(fetch.node_id)
    assert row.status == "done" and row.output == {"page": {"text": "x"}}
    assert [e["to_status"] for e in store.events(fetch.node_id)] == ["pending", "ready", "running", "done"]


def test_done_without_output_is_refused_by_code_and_by_schema(store):
    gid, _, _ = _page_graph(store)
    n = _by_type(store.nodes(gid), "recall")[0].node_id
    store.transition(n, "ready")
    store.start_node(n)
    with pytest.raises(ValueError):
        store.transition(n, "done")
    with pytest.raises(sqlite3.IntegrityError):
        with db.atomic(store.conn):
            store.conn.execute("UPDATE graph_nodes SET status='done' WHERE node_id=?", (n,))


def test_illegal_transitions_raise_and_roll_back(store):
    gid, _, _ = _page_graph(store)
    n = _by_type(store.nodes(gid), "recall")[0].node_id
    with pytest.raises(IllegalTransition):
        store.transition(n, "done", output={"memory": {}})
    assert store.node(n).status == "pending"
    assert len(store.events(n)) == 1


def test_extra_ops_roll_back_with_a_failed_transition(store):
    gid, _, _ = _page_graph(store)
    n = _by_type(store.nodes(gid), "recall")[0].node_id
    with pytest.raises(IllegalTransition):
        store.transition(n, "done", output={"memory": {}},
                         extra_ops=[("UPDATE graphs SET tokens_used = 999", ())])
    assert store.graph(gid).tokens_used == 0


def test_decision_frozen_at_start_cannot_be_revised(store):
    gid, _, _ = _page_graph(store)
    n = _by_type(store.nodes(gid), "extract")[0].node_id
    d1 = store.store_decision(graph_id=gid, node_id=n, node_type="extract", model_tag="m1",
                              models=["m1"], strategy="single", candidates=[], mode="fixed",
                              explore=False, reason="r")
    assert store.store_decision(graph_id=gid, node_id=n, node_type="extract", model_tag="m2",
                                models=["m2"], strategy="single", candidates=[], mode="fixed",
                                explore=False, reason="r") == d1  # no revise flag: keep
    d2 = store.store_decision(graph_id=gid, node_id=n, node_type="extract", model_tag="m2",
                              models=["m2"], strategy="single", candidates=[], mode="fixed",
                              explore=False, reason="swap", revise=True)
    assert store.live_decision(n)["decision_id"] == d2
    assert store.live_decision(n)["revised_from"] == d1
    store.transition(n, "ready")
    store.start_node(n)
    assert store.live_decision(n)["frozen"] == 1
    with pytest.raises(IllegalTransition):
        store.store_decision(graph_id=gid, node_id=n, node_type="extract", model_tag="m3",
                             models=["m3"], strategy="single", candidates=[], mode="fixed",
                             explore=False, reason="late", revise=True)


def test_resume_reset_returns_running_nodes_to_ready_with_an_event(store):
    gid, _, _ = _page_graph(store)
    n = _by_type(store.nodes(gid), "recall")[0].node_id
    store.transition(n, "ready")
    store.start_node(n)
    assert store.reset_running_for_resume(gid) == [n]
    assert store.node(n).status == "ready"
    assert '"resume"' in store.events(n)[-1]["detail"]


def test_samples_are_write_once(store):
    gid, _, _ = _page_graph(store)
    n = _by_type(store.nodes(gid), "extract")[0].node_id
    store.store_sample(n, 0, "m1", {"a": 1}, True, None)
    store.store_sample(n, 0, "m2", {"a": 2}, True, None)
    assert store.samples(n)[0]["output"] == {"a": 1}


# ---------------------------------------------------------------------------
# readiness (pure) on a stored graph
# ---------------------------------------------------------------------------


def _done(store, node_id, output):
    if store.node(node_id).status == "pending":
        store.transition(node_id, "ready")
    store.start_node(node_id)
    store.transition(node_id, "done", output=output)


def test_initial_sweep_readies_only_roots(store):
    gid, _, _ = _page_graph(store)
    sw = readiness.sweep(store.nodes(gid), store.edges(gid))
    types = sorted(store.nodes(gid)[n].node_type for n in sw.ready)
    assert types == ["fetch_page", "recall"]


def test_critique_failure_plans_a_paired_retry_with_the_reflection(store):
    gid, _, _ = _page_graph(store)
    nodes = store.nodes(gid)
    fetch, extract, crit = (_by_type(nodes, t)[0] for t in ("fetch_page", "extract", "critique"))
    _done(store, fetch.node_id, {"page": {"text": "Fees AUD 45,000"}, "evidence": {"text": "Fees AUD 45,000"}})
    _done(store, extract.node_id, {"subject": {"facts": []}})
    store.transition(crit.node_id, "ready")
    store.start_node(crit.node_id)
    store.transition(crit.node_id, "failed", output={"verdict": {"verdict": "fail", "reflection": "quote the fee"}},
                     error="critic rejected")
    plans = readiness.plan_retries(store.nodes(gid), store.edges(gid), Limits())
    assert len(plans) == 1
    exp_id, sub = plans[0]
    types = sorted(n.node_type for n in sub.nodes.values())
    assert types == ["critique", "extract"]
    new_extract = [n for n in sub.nodes.values() if n.node_type == "extract"][0]
    assert new_extract.params["reflection"] == "quote the fee" and new_extract.attempt == 2
    ok, v = store.apply_expansion(gid, sub, kind="retry", expansion_id=exp_id,
                                  counts_toward_limit=False, allow_existing_dst=True)
    assert ok, v
    # The same stored state plans nothing more (the retry exists now).
    assert readiness.plan_retries(store.nodes(gid), store.edges(gid), Limits()) == []
    # resolve_entity still waits: its facts chain has an open member.
    resolve = _by_type(store.nodes(gid), "resolve_entity")[0]
    assert resolve.node_id not in readiness.sweep(store.nodes(gid), store.edges(gid)).ready


def test_exhausted_retries_skip_the_downstream_chain(store):
    gid, _, _ = _page_graph(store)
    fetch = _by_type(store.nodes(gid), "fetch_page")[0]
    store.transition(fetch.node_id, "ready")
    store.start_node(fetch.node_id)
    store.transition(fetch.node_id, "failed", error="503")
    lim = Limits(max_attempts=1)
    assert readiness.plan_retries(store.nodes(gid), store.edges(gid), lim) == []
    sw = readiness.sweep(store.nodes(gid), store.edges(gid))
    skipped_types = {store.nodes(gid)[n].node_type for n in sw.skip}
    assert "extract" in skipped_types
    assert readiness.graph_outcome({**store.nodes(gid)}) is None  # recall etc. still pending


def _ns(**kw):
    base = dict(params={}, attempt=1, retry_of=None, prunable=False, output=None, error=None)
    base.update(kw)
    base.setdefault("retry_root", kw["node_id"])
    return SimpleNamespace(**base)


def test_fan_in_waits_while_a_planner_is_running():
    nodes = {
        "p": _ns(node_id="p", node_type="replan", status="running"),
        "d": _ns(node_id="d", node_type="diff", status="done", output={"changes": {"items": []}}),
        "s": _ns(node_id="s", node_type="summarize", status="pending"),
    }
    edges = [EdgeSpec("d", "s", "changes")]
    assert readiness.sweep(nodes, edges).ready == []
    nodes["p"].status = "done"
    assert readiness.sweep(nodes, edges).ready == ["s"]


def test_chain_value_is_the_highest_done_attempt():
    nodes = {
        "f": _ns(node_id="f", node_type="fetch_page", status="done", output={"page": "old"}),
        "f2": _ns(node_id="f2", node_type="fetch_page", status="done", attempt=2, retry_root="f",
                  output={"page": "new"}),
        "e": _ns(node_id="e", node_type="extract", status="pending"),
    }
    edges = [EdgeSpec("f", "e", "page"), EdgeSpec("f2", "e", "page")]
    assert readiness.resolve_inputs("e", nodes, edges) == {"page": "new"}


# ---------------------------------------------------------------------------
# property: an action is never ready before every gate ancestor is approved
# ---------------------------------------------------------------------------

STATUSES = ["pending", "ready", "running", "done", "failed", "blocked", "skipped"]


@st.composite
def notify_states(draw):
    nodes = {
        "s": _ns(node_id="s", node_type="summarize", status=draw(st.sampled_from(STATUSES)),
                 output={"digest": {"digest": "x"}}),
        "p": _ns(node_id="p", node_type="propose_action", status=draw(st.sampled_from(STATUSES)),
                 output={"proposal": {"body": "x"}}),
        "g": _ns(node_id="g", node_type="approval_gate", status=draw(st.sampled_from(STATUSES)),
                 output={"approval": {"status": "approved"}}),
        "x": _ns(node_id="x", node_type="execute_action", status="pending"),
    }
    for n in nodes.values():
        if n.status != "done":
            n.output = None
    edges = [EdgeSpec("s", "p", "digest"), EdgeSpec("p", "g", "proposal"), EdgeSpec("g", "x", "approval")]
    if draw(st.booleans()):  # a second gate on another path
        g2 = draw(st.sampled_from(STATUSES))
        nodes["g2"] = _ns(node_id="g2", node_type="approval_gate", status=g2,
                          output={"approval": {"status": "approved"}} if g2 == "done" else None)
        edges += [EdgeSpec("p", "g2", "proposal"), EdgeSpec("g2", "x", "approval")]
    return nodes, edges


def _gates_all_done(nodes):
    return all(n.status == "done" for n in nodes.values() if n.node_type == "approval_gate")


@settings(max_examples=500, deadline=None)
@given(notify_states())
def test_execute_action_ready_only_after_all_gates_done(case):
    nodes, edges = case
    if "x" in readiness.sweep(nodes, edges).ready:
        assert _gates_all_done(nodes)


def test_negative_control_without_the_gate_check_an_action_runs_early(monkeypatch):
    """A gate lookup that only reports approved gates (i.e. the 'every gate
    done' check is gone) must let an action run while another gate waits.
    Stubbing the lookup to [] would NOT do it: readiness fails closed when an
    action has no gate at all."""
    real = readiness.gate_ancestors
    monkeypatch.setattr(readiness, "gate_ancestors",
                        lambda node_id, nodes, edges: [g for g in real(node_id, nodes, edges)
                                                       if nodes[g].status == "done"])
    found = find(notify_states(),
                 lambda c: "x" in readiness.sweep(*c).ready and not _gates_all_done(c[0]),
                 settings=settings(max_examples=3000, deadline=None,
                                   suppress_health_check=list(HealthCheck)))
    assert not _gates_all_done(found[0])
