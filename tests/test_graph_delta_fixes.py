"""Tests for the delta review of the Phase E fixes."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from argus.graph import bandit, handlers, llm_nodes, procedural, registry
from argus.graph.fakes import FakeBackend
from argus.graph.health import HealthMonitor
from argus.graph.model import Limits
from argus.graph.router import Router, load_routing_config
from argus.graph.store import GraphStore, IllegalTransition
from argus.storage import db, lease

from tests.test_graph_engine import (MODELS, PAGE_V1, TARGET, URL, approve, env, fake_embed,  # noqa: F401
                                     make_engine, q)
from tests.test_graph_review_fixes import run


# -- High 1: malformed ports never raise ------------------------------------


@pytest.mark.parametrize("url", ["https://uni.edu:bad/x", "https://uni.edu:99999/x", "http://[::1:80/x"])
def test_same_site_refuses_malformed_urls_without_raising(url):
    assert handlers.same_site(url, "https://uni.edu/") is False


def test_mixed_scheme_default_ports_are_the_same_site():
    assert handlers.same_site("http://uni.edu/a", "https://uni.edu/b") is True
    assert handlers.same_site("http://uni.edu:8080/a", "https://uni.edu/b") is False


def test_a_planner_url_with_a_bad_port_is_dropped_not_fatal(env):
    db_path, outbox = env
    backend = FakeBackend()
    real = backend._answer
    backend._answer = lambda role, model, prompt: (
        {"thought": "x", "steps": [{"template": "check_page",
                                    "params": {"url": "http://127.0.0.1:99999/x", "fields": ["fees"]}}]}
        if role == "plan" else real(role, model, prompt))
    _, out = run(db_path, outbox, PAGE_V1, backend=backend)
    assert out.status in ("waiting_approval", "done")  # fell back to the target page


def test_search_skips_a_malformed_link():
    ex = SimpleNamespace(inputs={"page": {"url": URL, "links": [
        {"text": "fees", "href": "http://127.0.0.1:99999/fees"},
        {"text": "fees", "href": URL + "/fees"}]}}, params={"keywords": ["fees"]}, target=TARGET)
    r = asyncio.run(handlers._search(ex))
    assert r.status == "done" and r.output["url"] == URL + "/fees"


# -- M2: an action that fails after its claim is recorded as failed --------


def test_failed_action_after_claim_marks_the_row_failed(env):
    db_path, outbox = env

    class Broken:
        def execute(self, kind, payload, key):
            raise OSError("disk full")

    with db.connect(db_path) as conn:
        with db.atomic(conn):
            conn.execute("INSERT INTO runs (run_id, target_id, status) VALUES ('r', 't', 'running')")
            conn.execute("INSERT INTO approvals (idempotency_key, target_id, action_kind, proposed_action, "
                         "reasoning, status, created_run_id) VALUES ('k', 't', 'outbox_message', '{}', 'x', "
                         "'approved', 'r')")
        store = GraphStore(conn)
        ex = SimpleNamespace(inputs={"approval": {"idempotency_key": "k"}}, store=store,
                             services=SimpleNamespace(actions=Broken()), run_id="r",
                             node=SimpleNamespace(node_id="n"))
        r = asyncio.run(handlers._execute(ex))
        assert r.status == "failed"
        with db.atomic(conn):
            for sql, params in r.extra_ops:
                conn.execute(sql, params)
        row = conn.execute("SELECT status, result FROM approvals").fetchone()
    assert row["status"] == "failed" and json.loads(row["result"])["state"] == "failed"


# -- M6: rewards ----------------------------------------------------------


def test_reward_ops_are_idempotent_inside_one_transaction(tmp_path):
    from argus.storage.migrate import apply_migrations
    path = tmp_path / "r.db"
    apply_migrations(path)
    ops = bandit.reward_ops(node_type="extract", model="m", reward_key="n#0", source="validator",
                            reward=1.0, detail=None, prior_quality=0.9, strength=6)
    dup_new_arm = bandit.reward_ops(node_type="critique", model="m", reward_key="n#0", source="validator",
                                    reward=1.0, detail=None, prior_quality=0.5, strength=6)
    with db.connect(path) as conn:
        with db.atomic(conn):
            for sql, p in ops + ops:
                conn.execute(sql, p)
        row = conn.execute("SELECT alpha, beta, pulls FROM arm_stats WHERE node_type='extract'").fetchone()
        a0, b0 = bandit.seed_params(0.9, 6)
        assert (row["pulls"], row["alpha"], row["beta"]) == (1, pytest.approx(a0 + 1), pytest.approx(b0))
        # A new arm whose reward row already exists (same key, source, model):
        with db.atomic(conn):
            conn.execute("INSERT INTO arm_rewards (node_type, model_tag, node_id, source, reward) "
                         "VALUES ('critique', 'm', 'x#0', 'validator', 1.0)")
        dup = bandit.reward_ops(node_type="critique", model="m", reward_key="x#0", source="validator",
                                reward=1.0, detail=None, prior_quality=0.5, strength=6)
        with db.atomic(conn):
            for sql, p in dup:
                conn.execute(sql, p)
        assert conn.execute("SELECT pulls FROM arm_stats WHERE node_type='critique'").fetchone()["pulls"] == 0


def test_rewards_roll_back_with_a_failed_node_transition(env):
    db_path, outbox = env
    with db.connect(db_path) as conn:
        with db.atomic(conn):
            conn.execute("INSERT INTO runs (run_id, target_id, status) VALUES ('r', 't', 'running')")
        store = GraphStore(conn)
        from argus.graph import templates
        gid = store.create_graph(run_id="r", target_id="t", kind="watch", limits=Limits())
        sub, _ = templates.instantiate("check_page", gid, "t", {"url": URL, "fields": ["fees"]})
        store.apply_expansion(gid, sub, kind="initial", counts_toward_limit=False)
        node = next(n for n in store.nodes(gid).values() if n.node_type == "recall")
        ops = bandit.reward_ops(node_type="x", model="m", reward_key="k", source="validator", reward=1.0,
                                detail=None, prior_quality=0.5, strength=6)
        with pytest.raises(IllegalTransition):
            store.transition(node.node_id, "done", output={"memory": {}}, extra_ops=ops)
        assert conn.execute("SELECT COUNT(*) FROM arm_rewards").fetchone()[0] == 0


def test_fixed_mode_seeds_arms_from_the_profiler_prior(env):
    db_path, outbox = env
    with db.connect(db_path) as conn, db.atomic(conn):
        registry.write_profile(conn, MODELS[0], "plan", "quality", 0.9)
    run(db_path, outbox, PAGE_V1)
    row = q(db_path, "SELECT alpha, beta, pulls FROM arm_stats WHERE node_type='plan' AND model_tag=?", MODELS[0])
    a0, b0 = bandit.seed_params(0.9, load_routing_config()["prior_strength"])
    assert row and row[0]["alpha"] + row[0]["beta"] == pytest.approx(a0 + b0 + row[0]["pulls"])


def test_every_self_consistency_sample_is_rewarded(env):
    db_path, outbox = env
    eng, _ = make_engine(db_path, outbox, PAGE_V1)
    eng.cfg = {**eng.cfg, "self_consistency": {"types": ["critique"], "min_difficulty": 0.0,
                                               "samples": 3, "temperature": 0.4}}
    run(db_path, outbox, None, eng=eng)
    crit = q(db_path, "SELECT node_id FROM graph_nodes WHERE node_type='critique'")[0]["node_id"]
    keys = [r["node_id"] for r in q(db_path, "SELECT node_id FROM arm_rewards WHERE source='validator' "
                                    "AND node_id LIKE ?", crit + "#%")]
    assert sorted(keys) == [f"{crit}#{i}" for i in range(3)]


# -- M4, M5, M7 -------------------------------------------------------------


def test_rejected_plan_falls_back_in_one_expansion(env):
    db_path, outbox = env
    backend = FakeBackend()
    real = backend._answer
    backend._answer = lambda role, model, prompt: (
        {"thought": "x", "steps": [{"template": "check_page", "params": {"url": URL, "fields": ["fees"]}}] * 6}
        if role == "plan" else real(role, model, prompt))
    eng, _ = make_engine(db_path, outbox, PAGE_V1, backend=backend)
    eng.cfg = {**eng.cfg, "limits": {**eng.cfg["limits"], "max_nodes": 20}}
    run(db_path, outbox, None, eng=eng)
    plan = json.loads(q(db_path, "SELECT output FROM graph_nodes WHERE node_type='plan'")[0]["output"])
    assert any("planned graph rejected" in d for d in plan["dropped_steps"])
    rows = q(db_path, "SELECT kind, accepted FROM graph_expansions WHERE kind IN ('plan','engine')")
    assert rows == [{"kind": "plan", "accepted": 1}], "plan output and fallback are ONE expansion"


def test_run_closes_only_when_no_graph_is_open(env):
    db_path, _ = env
    with db.connect(db_path) as conn:
        with db.atomic(conn):
            conn.execute("INSERT INTO runs (run_id, target_id, status) VALUES ('r', 't', 'running')")
        store = GraphStore(conn)
        g1 = store.create_graph(run_id="r", target_id="t", kind="watch", limits=Limits())
        g2 = store.create_graph(run_id="r", target_id="t", kind="reflect", limits=Limits())
        store.finish_graph(g1, "done")
        assert conn.execute("SELECT status FROM runs").fetchone()[0] == "running"
        store.finish_graph(g2, "done")
        assert conn.execute("SELECT status FROM runs").fetchone()[0] == "done"


def test_templates_are_not_recorded_while_waiting_and_notify_counts_after(env):
    db_path, outbox = env
    _, out = run(db_path, outbox, PAGE_V1)
    assert out.status == "waiting_approval"
    assert q(db_path, "SELECT COUNT(*) AS n FROM template_outcomes")[0]["n"] == 0
    approve(db_path)
    eng2, _ = make_engine(db_path, outbox, PAGE_V1)
    eng2._bind_memory(out.run_id)
    asyncio.run(eng2.resume_run(out.run_id))
    notify = q(db_path, "SELECT success FROM template_outcomes WHERE template_id='notify'")
    assert notify == [{"success": 1}]


def _n(status, skip=None, root="r", attempt=1):
    return SimpleNamespace(status=status, skip_reason=skip, retry_root=root, attempt=attempt)


@pytest.mark.parametrize("members,ok", [
    ([_n("done")], True),
    ([_n("skipped", "empty digest; nothing to send")], True),
    ([_n("skipped", "approval rejected")], True),
    ([_n("skipped", "upstream input 'page' can never be satisfied")], False),
    ([_n("failed"), _n("skipped", "budget", attempt=2)], False),
    ([_n("failed"), _n("done", attempt=2)], True),
])
def test_template_success_rule(members, ok):
    assert procedural._succeeded(members) is ok


# -- M9 / independence -------------------------------------------------------


def test_trial_is_granted_once(env):
    db_path, _ = env
    with db.connect(db_path) as conn:
        with db.atomic(conn):
            conn.execute("INSERT INTO model_health (model_tag, status, degraded_at) "
                         "VALUES (?, 'degraded', '2000-01-01T00:00:00.000Z')", (MODELS[1],))
        r = Router(conn, [MODELS[1]], "fixed", load_routing_config(), baseline=MODELS[1])
        first = {c["model"]: c["health"] for c in r.candidates("extract", 0.5, set())}
        assert first[MODELS[1]] == "trial"
        r.decide(node_id="n1", node_type="extract", difficulty=0.5, attempt=1, resident=set())
        again = {c["model"]: c["health"] for c in r.candidates("extract", 0.5, set())}
        assert again[MODELS[1]] == "degraded"


def test_a_degraded_independent_critic_beats_the_author(env):
    db_path, _ = env
    a, b = MODELS
    with db.connect(db_path) as conn:
        h = HealthMonitor(conn, {"strikes": 1})
        h.record(b, tok_s=None, timed_out=True, ok=False)
        r = Router(conn, [a, b], "fixed", load_routing_config(), baseline=a)
        d = r.decide(node_id="c", node_type="critique", difficulty=0.5, attempt=1, resident={a}, exclude={a})
    assert d.model == b and "NO INDEPENDENT" not in d.reason


# -- lease on exceptions ------------------------------------------------------


def test_lease_is_released_when_the_engine_raises(env, monkeypatch):
    db_path, outbox = env
    eng, _ = make_engine(db_path, outbox, PAGE_V1, owner=lease.new_owner())
    run_id, gid = eng.start_run(TARGET)
    eng._bind_memory(run_id)
    monkeypatch.setattr(eng, "_housekeeping", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        asyncio.run(eng.run_graph(gid))
    assert q(db_path, "SELECT COUNT(*) AS n FROM run_leases")[0]["n"] == 0


# -- fence residue --------------------------------------------------------------


@pytest.mark.parametrize("evil", ["< /untrusted>", "</ untrusted>", "<\tuntrusted source=x>"])
def test_spaced_fence_tags_are_escaped(evil):
    assert "&lt;" in llm_nodes.fence(f"a {evil} b")
