"""Router, bandit and health: unit and property tests. No Ollama."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from argus.graph import bandit, registry
from argus.graph.health import HealthMonitor
from argus.graph.router import GraphRoutingState, Router, load_routing_config
from argus.storage import db
from argus.storage.migrate import apply_migrations

A, B, C = "a:latest", "b:latest", "c:latest"


@pytest.fixture
def conn(tmp_path):
    path = tmp_path / "r.db"
    apply_migrations(path)
    with db.connect(path) as c:
        yield c


def profile(conn, model, node_type="extract", quality=0.8, latency=4.0, valid=1.0, cold=5.0, tok=30.0):
    with db.atomic(conn):
        registry.write_profile(conn, model, node_type, "quality", quality)
        registry.write_profile(conn, model, node_type, "mean_latency_s", latency)
        registry.write_profile(conn, model, node_type, "json_valid_rate", valid)
        registry.write_profile(conn, model, "*", "cold_load_s", cold)
        registry.write_profile(conn, model, "*", "tok_s", tok)


def router(conn, mode="fixed", models=(A, B), **cfg_over):
    cfg = load_routing_config()
    cfg.update(cfg_over)
    return Router(conn, list(models), mode, cfg, baseline=A)


# ---------------------------------------------------------------------------
# scoring and decisions
# ---------------------------------------------------------------------------


def test_breakdown_of_every_candidate_is_returned(conn):
    profile(conn, A)
    profile(conn, B, quality=0.6)
    d = router(conn).decide(node_id="n1", node_type="extract", difficulty=0.5, attempt=1, resident=set())
    assert {c["model"] for c in d.candidates} == {A, B}
    for c in d.candidates:
        assert {"quality", "latency_s", "swap_s", "risk", "utility"} <= set(c)
    assert d.model == A


def test_swap_cost_can_outweigh_a_small_quality_edge(conn):
    profile(conn, A, quality=0.82, cold=8.0)
    profile(conn, B, quality=0.78, cold=8.0)
    r = router(conn)
    assert r.decide(node_id="n", node_type="extract", difficulty=0.5, attempt=1, resident=set()).model == A
    assert r.decide(node_id="n", node_type="extract", difficulty=0.5, attempt=1, resident={B}).model == B


def test_negative_control_without_the_swap_term_residency_is_ignored(conn):
    profile(conn, A, quality=0.82, cold=8.0)
    profile(conn, B, quality=0.78, cold=8.0)
    cfg = load_routing_config()
    cfg["weights"] = {**cfg["weights"], "swap": 0.0}
    r = Router(conn, [A, B], "fixed", cfg, baseline=A)
    assert r.decide(node_id="n", node_type="extract", difficulty=0.5, attempt=1, resident={B}).model == A


def test_coresident_load_costs_half_a_cold_load(conn):
    profile(conn, A, cold=6.0)
    profile(conn, B, cold=6.0)
    with db.atomic(conn):
        registry.write_coresidency(conn, A, B, True)
    cands = {c["model"]: c for c in router(conn).candidates("extract", 0.5, {B})}
    assert cands[A]["swap_s"] == pytest.approx(3.0) and cands[B]["swap_s"] == 0.0


def test_excluded_model_is_never_chosen_when_another_exists(conn):
    profile(conn, A, quality=0.99)
    profile(conn, B, quality=0.1)
    d = router(conn).decide(node_id="n", node_type="extract", difficulty=0.5, attempt=1,
                            resident={A}, exclude={A})
    assert d.model == B


def test_degraded_model_is_avoided(conn):
    profile(conn, A, quality=0.99)
    profile(conn, B, quality=0.5)
    h = HealthMonitor(conn, {"strikes": 2})
    h.record(A, tok_s=None, timed_out=True, ok=False)
    h.record(A, tok_s=None, timed_out=True, ok=False)
    assert router(conn).decide(node_id="n", node_type="extract", difficulty=0.5, attempt=1,
                               resident=set()).model == B


def test_baseline_mode_is_one_model_single_strategy(conn):
    profile(conn, B, quality=0.99)
    d = router(conn, mode="baseline").decide(node_id="n", node_type="propose_action", difficulty=0.9,
                                             attempt=1, resident={B}, default_strategy="cross_model")
    assert (d.model, d.models, d.strategy) == (A, [A], "single")


def test_cross_model_strategy_uses_two_different_models(conn):
    profile(conn, A, "propose_action")
    profile(conn, B, "propose_action")
    d = router(conn).decide(node_id="n", node_type="propose_action", difficulty=0.5, attempt=1,
                            resident=set(), default_strategy="cross_model")
    assert d.strategy == "cross_model" and len(set(d.models)) == 2


def test_retry_with_reflection_is_reflexion_retry(conn):
    profile(conn, A)
    d = router(conn).decide(node_id="n", node_type="extract", difficulty=0.5, attempt=2,
                            resident=set(), has_reflection=True)
    assert d.strategy == "reflexion_retry"


def test_weights_come_from_config_not_source(conn):
    profile(conn, A, quality=0.9, latency=30.0)
    profile(conn, B, quality=0.6, latency=1.0)
    quality_first = router(conn, weights={"quality": 5.0, "latency": 0.0, "swap": 0.0, "risk": 0.0})
    speed_first = router(conn, weights={"quality": 0.0, "latency": 5.0, "swap": 0.0, "risk": 0.0})
    assert quality_first.decide(node_id="n", node_type="extract", difficulty=0.5, attempt=1, resident=set()).model == A
    assert speed_first.decide(node_id="n", node_type="extract", difficulty=0.5, attempt=1, resident=set()).model == B


def test_revision_to_resident_only_when_loss_is_small(conn):
    profile(conn, A, quality=0.80, cold=6.0)
    profile(conn, B, quality=0.78, cold=6.0)
    r = router(conn)
    small = r.revise_to_resident(node_type="extract", difficulty=0.5,
                                 current={"model_tag": A}, resident={B})
    assert small is not None and small["model"] == B
    profile(conn, B, quality=0.05, cold=6.0)
    assert r.revise_to_resident(node_type="extract", difficulty=0.5,
                                current={"model_tag": A}, resident={B}) is None


# ---------------------------------------------------------------------------
# learning: bounded exploration, determinism, idempotent rewards
# ---------------------------------------------------------------------------


def test_learning_decisions_are_deterministic_per_node(conn):
    profile(conn, A, quality=0.6)
    profile(conn, B, quality=0.6)
    r = router(conn, mode="learning")
    first = [r.decide(node_id=f"n{i}", node_type="extract", difficulty=0.5, attempt=1,
                      resident=set(), state=GraphRoutingState(decisions=0)).model for i in range(20)]
    again = [r.decide(node_id=f"n{i}", node_type="extract", difficulty=0.5, attempt=1,
                      resident=set(), state=GraphRoutingState(decisions=0)).model for i in range(20)]
    assert first == again


@settings(max_examples=60, deadline=None)
@given(st.floats(0.05, 0.95), st.floats(0.05, 0.95), st.integers(5, 40))
def test_exploration_never_exceeds_the_cap(qa, qb, n):
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "x.db"
        apply_migrations(path)
        with db.connect(path) as c:
            profile(c, A, quality=qa)
            profile(c, B, quality=qb)
            r = router(c, mode="learning", explore_cap=0.2, prior_strength=1)
            state = GraphRoutingState()
            for i in range(n):
                r.decide(node_id=f"n{i}", node_type="extract", difficulty=0.5, attempt=1,
                         resident=set(), state=state)
            assert state.explorations <= 0.2 * state.decisions + 1e-9


def test_negative_control_uncapped_learning_explores_more(conn):
    profile(conn, A, quality=0.55)
    profile(conn, B, quality=0.5)
    capped, free = GraphRoutingState(), GraphRoutingState()
    r_cap = router(conn, mode="learning", explore_cap=0.0, prior_strength=1)
    r_free = router(conn, mode="learning", explore_cap=1.0, prior_strength=1)
    for i in range(40):
        r_cap.decide(node_id=f"n{i}", node_type="extract", difficulty=0.5, attempt=1, resident=set(), state=capped)
        r_free.decide(node_id=f"n{i}", node_type="extract", difficulty=0.5, attempt=1, resident=set(), state=free)
    assert capped.explorations == 0 and free.explorations > 0


def test_no_exploration_when_budget_is_low(conn):
    profile(conn, A, quality=0.55)
    profile(conn, B, quality=0.5)
    r = router(conn, mode="learning", explore_cap=1.0, prior_strength=1)
    state = GraphRoutingState(budget_remaining=0.1)
    for i in range(30):
        r.decide(node_id=f"n{i}", node_type="extract", difficulty=0.5, attempt=1, resident=set(), state=state)
    assert state.explorations == 0


def test_reward_is_applied_once_per_node_model_source(conn):
    for _ in range(3):
        bandit.apply_reward(conn, node_type="extract", model=A, node_id="n1", source="validator", reward=1.0)
    row = conn.execute("SELECT alpha, beta, pulls FROM arm_stats WHERE model_tag = ?", (A,)).fetchone()
    a0, b0 = bandit.seed_params(0.5, 6.0)
    assert (row["alpha"], row["beta"], row["pulls"]) == (pytest.approx(a0 + 1), pytest.approx(b0), 1)


def test_learning_moves_the_choice_toward_the_rewarded_model(conn):
    profile(conn, A, quality=0.5)
    profile(conn, B, quality=0.5)
    for i in range(30):
        bandit.apply_reward(conn, node_type="extract", model=B, node_id=f"x{i}", source="validator", reward=1.0)
        bandit.apply_reward(conn, node_type="extract", model=A, node_id=f"x{i}", source="validator", reward=0.0)
    r = router(conn, mode="learning", explore_cap=0.0)
    picks = [r.decide(node_id=f"n{i}", node_type="extract", difficulty=0.5, attempt=1, resident=set(),
                      state=GraphRoutingState()).model for i in range(10)]
    assert picks.count(B) == 10


# ---------------------------------------------------------------------------
# health: half-open circuit breaker
# ---------------------------------------------------------------------------


def test_slow_model_degrades_then_gets_one_trial_after_cooldown(conn):
    profile(conn, A, tok=30.0)
    clock = {"t": datetime(2026, 1, 1, tzinfo=timezone.utc)}
    h = HealthMonitor(conn, {"degrade_ratio": 0.35, "strikes": 2, "cooldown_s": 60}, now=lambda: clock["t"])
    assert h.record(A, tok_s=5.0, timed_out=False, ok=True) == "healthy"
    assert h.record(A, tok_s=5.0, timed_out=False, ok=True) == "degraded"
    assert h.status(A) == "degraded"
    clock["t"] += timedelta(seconds=61)
    assert h.status(A) == "trial"
    assert h.record(A, tok_s=28.0, timed_out=False, ok=True) == "healthy"


def test_failed_trial_reopens_the_breaker(conn):
    profile(conn, A, tok=30.0)
    clock = {"t": datetime(2026, 1, 1, tzinfo=timezone.utc)}
    h = HealthMonitor(conn, {"strikes": 2, "cooldown_s": 60}, now=lambda: clock["t"])
    h.record(A, tok_s=None, timed_out=True, ok=False)
    h.record(A, tok_s=None, timed_out=True, ok=False)
    clock["t"] += timedelta(seconds=61)
    assert h.record(A, tok_s=None, timed_out=True, ok=False) == "degraded"
    assert h.status(A) == "degraded"
