"""Router: pick a model and an execution strategy for one LLM node.

This is a scheduling problem, not "best model per node" (ARGUS_BUILD 6.2):
only one chat model fits in VRAM, a swap costs a cold load, so every
candidate's score carries its swap cost against what is resident NOW.

    quality   learning: Thompson sample from Beta(arm); fixed: prior mean
    latency   measured mean call seconds for this node type (profile)
    swap      0 if resident; half the cold load if it can load beside what is
              resident (co-residency matrix: nothing gets evicted); else the
              full cold load
    risk      (1 - json_valid_rate) + 0.5 per recent consecutive timeout
    utility   wq*(0.5+difficulty)*quality - wl*latency/scale - ws*swap/scale - wr*risk

Every candidate's components are returned and stored in
routing_decisions.candidates, so "why this model" is one SQL query.

Modes: 'baseline' (everything on the configured baseline model, strategy
single: the Sprint 1 design), 'fixed' (profiler priors, no learning),
'learning' (Thompson sampling with bounded exploration).
"""

from __future__ import annotations

import hashlib
import random
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import yaml

from argus import config
from argus.graph import bandit, registry
from argus.graph.health import HealthMonitor


def load_routing_config(path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or (config.config_dir() / "routing.yaml")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass
class Decision:
    model: str
    models: List[str]
    strategy: str
    candidates: List[Dict[str, Any]]
    explore: bool
    reason: str
    mode: str


@dataclass
class GraphRoutingState:
    """Per-graph counters the bounded-exploration rule needs."""

    decisions: int = 0
    explorations: int = 0
    budget_remaining: float = 1.0


def _rng(node_id: str, salt: str) -> random.Random:
    seed = int(hashlib.sha256(f"{node_id}\x00{salt}".encode()).hexdigest()[:16], 16)
    return random.Random(seed)


class Router:
    def __init__(self, conn: sqlite3.Connection, models: List[str], mode: str,
                 cfg: Optional[Dict[str, Any]] = None, baseline: Optional[str] = None):
        if mode not in ("baseline", "fixed", "learning"):
            raise ValueError(f"unknown routing mode {mode!r}")
        self.conn = conn
        self.models = list(models)
        self.mode = mode
        self.cfg = cfg if cfg is not None else load_routing_config()
        self.baseline = config.normalize_tag(baseline or config.model_for_role("planner"))
        self.health = HealthMonitor(conn, self.cfg.get("health", {}))

    # ------------------------------------------------------------ scoring

    def _p(self, model: str, node_type: str, metric: str, default_key: str) -> float:
        d = self.cfg.get("defaults", {}).get(default_key)
        v = registry.profile(self.conn, model, node_type, metric)
        if v is None and node_type != "*":
            v = registry.profile(self.conn, model, "*", metric)
        return float(v if v is not None else d)

    def score(self, model: str, node_type: str, difficulty: float, resident: Set[str],
              coresident: Dict, rng: Optional[random.Random], learning: bool) -> Dict[str, Any]:
        w = self.cfg.get("weights", {})
        scale = float(self.cfg.get("latency_scale_s", 10.0))
        prior_q = self._p(model, node_type, "quality", "quality")
        strength = float(self.cfg.get("prior_strength", 6))
        if learning:
            alpha, beta = bandit.ensure_arm(self.conn, node_type, model, prior_q, strength)
        else:
            alpha, beta = bandit.seed_params(prior_q, strength)
        q_mean = bandit.mean(alpha, beta)
        q = bandit.sample(alpha, beta, rng) if (learning and rng is not None) else q_mean
        latency = self._p(model, node_type, "mean_latency_s", "mean_latency_s")
        cold = self._p(model, "*", "cold_load_s", "cold_load_s")
        if model in resident:
            swap = 0.0
        elif resident and all(coresident.get((model, r), False) for r in resident):
            swap = cold * 0.5
        else:
            swap = cold
        valid = self._p(model, node_type, "json_valid_rate", "json_valid_rate")
        hrow = self.conn.execute(
            "SELECT consecutive_timeouts FROM model_health WHERE model_tag = ?", (model,)
        ).fetchone()
        risk = (1.0 - valid) + 0.5 * (hrow["consecutive_timeouts"] if hrow else 0)
        wq = float(w.get("quality", 1.0)) * (0.5 + float(difficulty))
        utility = (wq * q - float(w.get("latency", 0.25)) * latency / scale
                   - float(w.get("swap", 0.35)) * swap / scale - float(w.get("risk", 0.6)) * risk)
        greedy = (wq * q_mean - float(w.get("latency", 0.25)) * latency / scale
                  - float(w.get("swap", 0.35)) * swap / scale - float(w.get("risk", 0.6)) * risk)
        return {"model": model, "quality": round(q, 4), "quality_mean": round(q_mean, 4),
                "prior_quality": round(prior_q, 4), "alpha": round(alpha, 3), "beta": round(beta, 3),
                "latency_s": round(latency, 3), "swap_s": round(swap, 3), "risk": round(risk, 4),
                "utility": round(utility, 4), "greedy_utility": round(greedy, 4),
                "resident": model in resident, "health": self.health.routable(model)}

    def candidates(self, node_type: str, difficulty: float, resident: Set[str],
                   exclude: Iterable[str] = (), node_id: str = "", learning: bool = False
                   ) -> List[Dict[str, Any]]:
        excluded = set(exclude)
        co = registry.coresidency(self.conn)
        rng = _rng(node_id, "thompson") if learning else None
        out = []
        for m in self.models:
            c = self.score(m, node_type, difficulty, resident, co, rng, learning)
            c["excluded"] = ("independence" if m in excluded else
                             "degraded" if c["health"] == "degraded" else None)
            out.append(c)
        return sorted(out, key=lambda c: (-c["utility"], c["model"]))

    # ------------------------------------------------------------ decide

    def decide(self, *, node_id: str, node_type: str, difficulty: float, attempt: int,
               resident: Set[str], exclude: Iterable[str] = (),
               state: Optional[GraphRoutingState] = None, has_reflection: bool = False,
               default_strategy: str = "single") -> Decision:
        state = state or GraphRoutingState()
        exclude = set(exclude)
        if self.mode == "baseline":
            cands = [{"model": self.baseline, "reason": "baseline mode"}]
            # Baseline is the Sprint 1 design: one model, one call. The only
            # thing it keeps is the hard rule that a critic never grades
            # itself... which the single-model design cannot honour; recorded.
            return Decision(self.baseline, [self.baseline], "single", cands, False,
                            "baseline: single model, single strategy", "baseline")

        learning = self.mode == "learning"
        cands = self.candidates(node_type, difficulty, resident, exclude, node_id, learning)
        usable = [c for c in cands if not c["excluded"]] or [c for c in cands if c["excluded"] != "independence"]
        independence_note = ""
        if not usable:
            # Only when the WHOLE pool is the subject's own model (a one-model
            # pool). Recorded in the decision, and the critic reward is never
            # applied when author == critic (engine._reward_ops). D-022.
            usable = cands
            independence_note = " [NO INDEPENDENT MODEL IN POOL: critic shares the author's model]"
        explore = False
        chosen = usable[0]
        if learning:
            greedy = max(usable, key=lambda c: (c["greedy_utility"], c["model"]))
            cap = float(self.cfg.get("explore_cap", 0.2))
            allowed = (state.budget_remaining >= float(self.cfg.get("explore_min_budget", 0.3))
                       and (state.explorations + 1) <= cap * max(state.decisions + 1, 1))
            if chosen["model"] != greedy["model"]:
                if allowed:
                    explore = True
                else:
                    chosen = greedy
        state.decisions += 1
        state.explorations += int(explore)

        strategy, models = self._strategy(node_type, difficulty, attempt, has_reflection,
                                          default_strategy, chosen, usable)
        by_model = {c["model"]: c for c in cands}
        for m in models:  # every model this decision may call, not only the first
            if by_model.get(m, {}).get("health") == "trial":
                self.health.grant_trial(m)  # its ONE half-open call
        reason = (f"{chosen['model']} utility {chosen['utility']:.3f} "
                  f"(q {chosen['quality']:.2f}, lat {chosen['latency_s']:.1f}s, swap {chosen['swap_s']:.1f}s, "
                  f"risk {chosen['risk']:.2f})" + (" [explore]" if explore else "") + independence_note)
        return Decision(chosen["model"], models, strategy, cands, explore, reason, self.mode)

    def _strategy(self, node_type, difficulty, attempt, has_reflection, default, chosen, usable):
        others = [c["model"] for c in usable if c["model"] != chosen["model"]]
        if default == "cross_model":
            if others:
                return "cross_model", [chosen["model"], others[0]]
            return "single", [chosen["model"]]
        if attempt > 1 and has_reflection:
            return "reflexion_retry", [chosen["model"]]
        sc = self.cfg.get("self_consistency", {})
        if node_type in sc.get("types", []) and difficulty >= float(sc.get("min_difficulty", 1.1)):
            return "self_consistency", [chosen["model"]]
        cas = self.cfg.get("cascade", {})
        if node_type in cas.get("types", []):
            adequate = [c for c in usable if c["quality_mean"] >= float(cas.get("adequate_quality", 0.7))]
            if adequate:
                cheapest = min(adequate, key=lambda c: (c["latency_s"] + c["swap_s"], c["model"]))
                if cheapest["model"] != chosen["model"]:
                    return "cascade", [cheapest["model"], chosen["model"]]
        return "single", [chosen["model"]]

    def revise_to_resident(self, *, node_type: str, difficulty: float, current: Dict[str, Any],
                           resident: Set[str], exclude: Iterable[str] = ()) -> Optional[Dict[str, Any]]:
        """If switching a not-yet-started node to a resident model loses at
        most `revise_max_loss` utility, return that candidate (else None)."""
        if self.mode == "baseline" or not resident or current["model_tag"] in resident:
            return None
        cands = {c["model"]: c for c in self.candidates(node_type, difficulty, resident, exclude)}
        here = cands.get(current["model_tag"])
        options = [cands[m] for m in resident if m in cands and not cands[m]["excluded"]]
        if here is None or not options:
            return None
        best = max(options, key=lambda c: (c["greedy_utility"], c["model"]))
        loss = here["greedy_utility"] - best["greedy_utility"]
        if loss <= float(self.cfg.get("revise_max_loss", 0.15)):
            best = dict(best)
            best["revision_loss"] = round(loss, 4)
            return best
        return None
