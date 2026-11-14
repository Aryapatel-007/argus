"""Graph engine end to end, with a fake model, a static browser and the real
memory backend (deterministic fake embeddings). No Ollama needed.

Trajectory assertions read node_events, routing_decisions, llm_calls and
approvals back from SQLite (argus-agentic-testing skill).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from argus import embeddings
from argus.graph import engine as engine_mod
from argus.graph.engine import Engine
from argus.graph.fakes import FakeBackend
from argus.graph.services import ActionExecutor, DirectMemory, Services, StaticBrowser
from argus.graph.store import GraphStore
from argus.storage import db
from argus.storage.migrate import apply_migrations

from tests.test_audit_fixes import _unit

URL = "http://127.0.0.1:8765/uni/master-ai"
PAGE_V1 = {"title": "Master of AI", "text": (
    "Master of Artificial Intelligence. Duration: 2 years full-time. "
    "Intakes: February and July each year. The annual tuition fee for 2027 is AUD 45,000. "
    "Entry requires a bachelor degree with a WAM of 65.")}
PAGE_V2 = {"title": "Master of AI", "text": PAGE_V1["text"].replace("45,000", "48,500")}
TARGET = {"id": "uni", "name": "Uni Master of AI", "url": URL,
          "extract": ["intake_dates", "duration", "fees", "entry_requirements"], "notify": True}
MODELS = ["model-a:latest", "model-b:latest"]


@pytest.fixture(autouse=True)
def fake_embed(monkeypatch):
    async def embed(text, *, run_id, conn=None, task_id=None, step_index=None):
        return _unit(text)
    monkeypatch.setattr(embeddings, "embed", embed)


@pytest.fixture
def env(tmp_path):
    path = tmp_path / "argus.db"
    apply_migrations(path)
    return path, tmp_path / "outbox"


def make_engine(db_path, outbox, page, backend=None, mode="fixed", owner="test-owner", run_id="probe"):
    backend = backend or FakeBackend()
    memory_holder = {}

    class LazyMemory:
        """DirectMemory needs a run id that exists; bind it per call."""

        async def _m(self):
            return memory_holder["m"]

        async def list_current_facts(self, target_id):
            return await memory_holder["m"].list_current_facts(target_id)

        async def recall(self, query, target_id, limit=5):
            return await memory_holder["m"].recall(query, target_id, limit)

        async def remember(self, **kw):
            return await memory_holder["m"].remember(**kw)

    services = Services(backend=backend, memory=LazyMemory(), browser=StaticBrowser({URL: page}),
                        actions=ActionExecutor(outbox))
    eng = Engine(db_path, services, mode=mode, models=MODELS, baseline=MODELS[0], owner=owner)

    def bind(run_id):
        memory_holder["m"] = DirectMemory(db_path, run_id)
    eng._bind_memory = bind
    return eng, backend


def run_once(db_path, outbox, page, **kw):
    eng, backend = make_engine(db_path, outbox, page, **kw)
    run_id, graph_id = eng.start_run(TARGET)
    eng._bind_memory(run_id)
    outcome = asyncio.run(eng.run_graph(graph_id))
    return eng, backend, outcome


def q(db_path, sql, *params):
    with db.connect(db_path) as conn:
        return [dict(r) for r in conn.execute(sql, params)]


def approve(db_path, status="approved"):
    with db.connect(db_path) as conn, db.atomic(conn):
        conn.execute("UPDATE approvals SET status = ?, decided_by = 'test', "
                     "resolved_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE status = 'pending'", (status,))


# ---------------------------------------------------------------------------


def test_first_run_adds_facts_digests_and_waits_for_approval(env):
    db_path, outbox = env
    eng, backend, out = run_once(db_path, outbox, PAGE_V1)
    assert out.status == "waiting_approval", out
    facts = q(db_path, "SELECT entity_key, change_kind FROM facts WHERE is_current = 1 ORDER BY entity_key")
    assert {f["entity_key"] for f in facts} == set(TARGET["extract"])
    assert all(f["change_kind"] == "added" for f in facts)
    assert out.digest and "45,000" in out.digest
    approvals = q(db_path, "SELECT status FROM approvals")
    assert [a["status"] for a in approvals] == ["pending"]
    assert not outbox.exists() or not list(outbox.iterdir()), "nothing may leave before approval"


def test_every_llm_node_had_a_frozen_decision_before_it_ran(env):
    db_path, outbox = env
    run_once(db_path, outbox, PAGE_V1)
    rows = q(db_path, """
        SELECT n.node_id, n.node_type, n.started_at, d.created_at AS decided, d.frozen, d.model_tag
        FROM graph_nodes n JOIN routing_decisions d ON d.node_id = n.node_id AND d.superseded = 0
        WHERE n.started_at IS NOT NULL""")
    assert rows
    for r in rows:
        assert r["frozen"] == 1 and r["decided"] <= r["started_at"], r
    llm_nodes = q(db_path, "SELECT node_id FROM graph_nodes WHERE node_type IN "
                  "('plan','extract','critique','write_fact','summarize','propose_action','resolve_entity') "
                  "AND started_at IS NOT NULL")
    assert {r["node_id"] for r in llm_nodes} <= {r["node_id"] for r in rows}


def test_critic_never_uses_the_extractors_model(env):
    db_path, outbox = env
    run_once(db_path, outbox, PAGE_V1)
    pairs = q(db_path, """
        SELECT dc.model_tag AS critic, ds.models AS subject_models
        FROM graph_edges e
        JOIN routing_decisions dc ON dc.node_id = e.dst AND dc.superseded = 0
        JOIN routing_decisions ds ON ds.node_id = e.src AND ds.superseded = 0
        WHERE e.channel = 'subject'""")
    assert pairs
    for p in pairs:
        assert p["critic"] not in json.loads(p["subject_models"])


def test_approved_action_executes_exactly_once_on_resume(env):
    db_path, outbox = env
    eng, _, out = run_once(db_path, outbox, PAGE_V1)
    approve(db_path)
    eng2, _ = make_engine(db_path, outbox, PAGE_V1)
    eng2._bind_memory(out.run_id)
    outcomes = asyncio.run(eng2.resume_run(out.run_id))
    assert outcomes[0].status == "done"
    files = list(outbox.iterdir())
    assert len(files) == 1
    assert q(db_path, "SELECT status FROM approvals")[0]["status"] == "executed"
    # Resuming again does nothing more.
    eng3, _ = make_engine(db_path, outbox, PAGE_V1)
    eng3._bind_memory(out.run_id)
    assert asyncio.run(eng3.resume_run(out.run_id)) == []
    assert len(list(outbox.iterdir())) == 1


def test_rejected_action_never_executes(env):
    db_path, outbox = env
    _, _, out = run_once(db_path, outbox, PAGE_V1)
    approve(db_path, "rejected")
    eng2, _ = make_engine(db_path, outbox, PAGE_V1)
    eng2._bind_memory(out.run_id)
    outcomes = asyncio.run(eng2.resume_run(out.run_id))
    assert outcomes[0].status == "done"
    assert not outbox.exists() or not list(outbox.iterdir())
    statuses = {r["node_type"]: r["status"] for r in q(
        db_path, "SELECT node_type, status FROM graph_nodes WHERE node_type IN ('approval_gate','execute_action')")}
    assert statuses == {"approval_gate": "skipped", "execute_action": "skipped"}


def test_unchanged_page_is_silent_and_queues_nothing(env):
    db_path, outbox = env
    run_once(db_path, outbox, PAGE_V1)
    approve(db_path, "rejected")
    _, _, out2 = run_once(db_path, outbox, PAGE_V1)
    assert out2.status == "done"
    assert not out2.digest
    assert len(q(db_path, "SELECT 1 FROM approvals")) == 1  # only the first run's
    kinds = q(db_path, "SELECT change_kind FROM facts")
    assert len(kinds) == 4, "an unchanged page must not create new fact versions"


def test_changed_fee_is_modified_and_digested(env):
    db_path, outbox = env
    run_once(db_path, outbox, PAGE_V1)
    _, _, out2 = run_once(db_path, outbox, PAGE_V2)
    fees = q(db_path, "SELECT content, change_kind, version FROM facts WHERE entity_key='fees' ORDER BY version")
    assert [f["change_kind"] for f in fees] == ["added", "modified"]
    assert "48,500" in fees[-1]["content"]
    assert "48,500" in out2.digest and "45,000" in out2.digest


def test_invented_number_is_caught_by_the_validator_before_any_critic(env):
    """Deterministic checks run first (ARGUS_BUILD 6.4): the critic never
    sees an extract whose numbers are not on the page."""
    db_path, outbox = env
    backend = FakeBackend()
    backend.wrong_number_first = {m: 1 for m in MODELS}
    run_once(db_path, outbox, PAGE_V1, backend=backend)
    extracts = q(db_path, "SELECT attempt, status, params FROM graph_nodes WHERE node_type='extract' ORDER BY attempt")
    assert [(e["attempt"], e["status"]) for e in extracts] == [(1, "failed"), (2, "done")]
    assert "99999" in json.loads(extracts[1]["params"])["reflection"]
    assert [r["status"] for r in q(db_path, "SELECT status FROM graph_nodes WHERE node_type='critique'")] == ["done"]
    fee = q(db_path, "SELECT content FROM facts WHERE entity_key='fees'")[0]["content"]
    assert "45,000" in fee


def test_critique_rejection_retries_extract_with_the_reflection(env):
    db_path, outbox = env
    backend = FakeBackend()
    backend.mislabel_first = {m: 1 for m in MODELS}
    _, _, out = run_once(db_path, outbox, PAGE_V1, backend=backend)
    extracts = q(db_path, "SELECT node_id, attempt, status, params FROM graph_nodes "
                 "WHERE node_type='extract' ORDER BY attempt")
    assert [e["attempt"] for e in extracts] == [1, 2]
    assert "different field" in json.loads(extracts[1]["params"])["reflection"]
    crit = q(db_path, "SELECT attempt, status FROM graph_nodes WHERE node_type='critique' ORDER BY attempt")
    assert [(c["attempt"], c["status"]) for c in crit] == [(1, "failed"), (2, "done")]
    dur = q(db_path, "SELECT content FROM facts WHERE entity_key='duration'")[0]["content"]
    assert "2 years" in dur, "the mislabelled fact must never reach memory"
    rewards = q(db_path, "SELECT source, reward FROM arm_rewards WHERE source='critic' ORDER BY reward_id")
    assert [r["reward"] for r in rewards] == [0.0, 1.0]


def test_invalid_json_is_retried_and_logged(env):
    db_path, outbox = env
    backend = FakeBackend()
    backend.fail_first = {("summarize", m): 1 for m in MODELS}
    _, _, out = run_once(db_path, outbox, PAGE_V1, backend=backend)
    assert out.digest
    bad = q(db_path, "SELECT ok, error FROM llm_calls WHERE role='summarize' ORDER BY created_at")
    assert bad[0]["ok"] == 0 and "schema" in bad[0]["error"]


def test_baseline_mode_uses_one_model_and_no_learning(env):
    db_path, outbox = env
    run_once(db_path, outbox, PAGE_V1, mode="baseline")
    models = {r["model_tag"] for r in q(db_path, "SELECT DISTINCT model_tag FROM llm_calls")}
    assert models == {MODELS[0]}
    assert q(db_path, "SELECT COUNT(*) AS n FROM arm_rewards")[0]["n"] == 0
    assert {r["strategy"] for r in q(db_path, "SELECT strategy FROM routing_decisions")} == {"single"}


def test_tiny_budget_prunes_optional_work_but_never_a_gate(env, monkeypatch):
    db_path, outbox = env
    eng, backend = make_engine(db_path, outbox, PAGE_V1)
    eng.cfg = {**eng.cfg, "budget": {"seconds": 1e-9, "tokens": 1, "prune_fraction": 0.99}}
    run_id, gid = eng.start_run(TARGET)
    eng._bind_memory(run_id)
    asyncio.run(eng.run_graph(gid))
    pruned = q(db_path, "SELECT node_type FROM graph_nodes WHERE skip_reason = 'budget'")
    assert all(p["node_type"] not in ("approval_gate", "execute_action") for p in pruned)


def test_llm_calls_carry_node_strategy_and_decision(env):
    db_path, outbox = env
    run_once(db_path, outbox, PAGE_V1)
    rows = q(db_path, "SELECT node_id, strategy, decision_id, sample_index, think_mode FROM llm_calls")
    assert rows and all(r["node_id"] and r["strategy"] and r["decision_id"] for r in rows)
    assert {r["think_mode"] for r in rows} == {"False"}
