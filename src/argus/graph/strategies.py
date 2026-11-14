"""Execution strategies for LLM nodes (ARGUS_BUILD 6.6).

    single           one call on the routed model
    reflexion_retry  one call; the node's params carry the critic's or the
                     validator's reflection from the failed attempt, which the
                     prompt renders (the retry itself is a new node, D-013)
    cascade          cheapest adequate model first; the stronger model only if
                     the deterministic validator rejects the first answer
    cross_model      two different models; both must be valid AND agree
    self_consistency N seeded samples on one model; the answer most others
                     agree with wins; disagreement is reported as uncertainty

Plus one escalation that any strategy can trigger on `extract`: a fact whose
confidence is below `low_confidence` must be confirmed by a second, different
model, or it is dropped (ARGUS_BUILD 6.6 cross_model rule for low-confidence
facts).

Every sample is stored in node_samples BEFORE the next one starts, and a
stored sample is reused instead of re-called (crash-resume: never regenerate).
Every call holds the GPU lane for its duration only.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from argus.graph import llm_nodes
from argus.graph.llm import CallResult, LogContext


@dataclass
class Attempt:
    index: int
    model: str
    parsed: Optional[Dict[str, Any]]
    problems: List[str]
    error: Optional[str]
    reused: bool = False
    eval_count: int = 0

    @property
    def valid(self) -> bool:
        return self.parsed is not None and not self.problems


@dataclass
class StrategyResult:
    ok: bool
    parsed: Optional[Dict[str, Any]]
    model: Optional[str]
    attempts: List[Attempt] = field(default_factory=list)
    error: Optional[str] = None
    uncertainty: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(a.eval_count for a in self.attempts if not a.reused)


def seed_for(node_id: str, index: int) -> int:
    return int(hashlib.sha256(f"{node_id}:{index}".encode()).hexdigest()[:8], 16)


class StrategyRunner:
    """Bound to one node execution. `call` is the only path to a model."""

    def __init__(self, *, ex: Any, spec: llm_nodes.LLMNode, decision: Dict[str, Any]):
        self.ex = ex  # engine NodeExecution: store, node, inputs, params, target, services...
        self.spec = spec
        self.decision = decision
        self.stored = ex.store.samples(ex.node.node_id)

    async def call(self, index: int, model: str, *, temperature: float = 0.0,
                   seed: Optional[int] = None, inputs: Optional[Dict[str, Any]] = None,
                   params: Optional[Dict[str, Any]] = None) -> Attempt:
        inputs = inputs if inputs is not None else self.ex.inputs
        params = params if params is not None else self.ex.params
        if index in self.stored:
            s = self.stored[index]
            problems = self.spec.validate(s["output"], inputs, params) if s["output"] is not None else ["no output"]
            return Attempt(index, s["model_tag"], s["output"], problems if s["ok"] else (problems or [s["error"] or "failed"]),
                           s["error"], reused=True)
        messages = self.spec.messages(inputs, params, self.ex.target)
        log = LogContext(run_id=self.ex.run_id, role=self.ex.role, conn=self.ex.store.conn,
                         graph_id=self.ex.graph_id, node_id=self.ex.node.node_id,
                         strategy=self.decision["strategy"], sample_index=index,
                         decision_id=self.decision["decision_id"], prompt_file=self.spec.prompt_file)
        async with self.ex.gpu:
            result: CallResult = await self.ex.services.backend.chat(
                model=model, messages=messages, schema=self.spec.schema, log=log,
                temperature=temperature, seed=seed)
            self.ex.on_model_used(model, result)
        problems = self.spec.validate(result.parsed, inputs, params) if result.parsed is not None else \
            [result.error or "no parsed output"]
        ok = result.parsed is not None and not problems
        self.ex.store.store_sample(self.ex.node.node_id, index, model, result.parsed, ok,
                                   None if ok else "; ".join(problems)[:2000])
        return Attempt(index, model, result.parsed, problems if result.parsed is not None else problems,
                       result.error, eval_count=result.eval_count or 0)

    # ------------------------------------------------------------------ run

    async def run(self) -> StrategyResult:
        strategy = self.decision["strategy"]
        models = self.decision["models"]
        if strategy in ("single", "reflexion_retry"):
            a = await self.call(0, models[0])
            res = StrategyResult(a.valid, a.parsed if a.valid else None, a.model, [a],
                                 None if a.valid else "; ".join(a.problems)[:1000])
        elif strategy == "cascade":
            a = await self.call(0, models[0])
            if a.valid or len(models) < 2:
                res = StrategyResult(a.valid, a.parsed if a.valid else None, a.model, [a],
                                     None if a.valid else "; ".join(a.problems)[:1000])
                if a.valid:
                    res.notes.append("cascade: cheap model accepted")
            else:
                b = await self.call(1, models[1])
                res = StrategyResult(b.valid, b.parsed if b.valid else None, b.model, [a, b],
                                     None if b.valid else "; ".join(b.problems)[:1000],
                                     notes=["cascade: escalated after validator rejection"])
        elif strategy == "cross_model":
            a = await self.call(0, models[0])
            b = await self.call(1, models[1]) if len(models) > 1 else a
            agree = a.valid and b.valid and self.spec.agree(a.parsed, b.parsed)
            err = None if agree else (
                "cross_model: " + ("models disagree" if a.valid and b.valid else
                                   "; ".join(a.problems + b.problems)[:1000]))
            res = StrategyResult(agree, a.parsed if agree else None, a.model, [a, b], err,
                                 uncertainty=0.0 if agree else 1.0)
        elif strategy == "self_consistency":
            cfg = self.ex.routing_cfg.get("self_consistency", {})
            n = int(cfg.get("samples", 3))
            temp = float(cfg.get("temperature", 0.4))
            attempts = [await self.call(i, models[0], temperature=temp, seed=seed_for(self.ex.node.node_id, i))
                        for i in range(n)]
            valid = [a for a in attempts if a.valid]
            best, support = None, 0
            for a in valid:
                s = sum(1 for b in valid if self.spec.agree(a.parsed, b.parsed))
                if s > support:
                    best, support = a, s
            ok = best is not None and support >= (n // 2 + 1)
            res = StrategyResult(ok, best.parsed if ok else None, models[0], attempts,
                                 None if ok else "self_consistency: no majority",
                                 uncertainty=1.0 - support / n)
        else:
            raise ValueError(f"unknown strategy {strategy!r}")

        if res.ok and self.ex.node.node_type == "extract":
            await self._confirm_low_confidence(res)
        return res

    async def _confirm_low_confidence(self, res: StrategyResult) -> None:
        threshold = float(self.ex.routing_cfg.get("low_confidence", 0.6))
        low = [f for f in res.parsed.get("facts", []) if float(f.get("confidence", 1.0)) < threshold]
        if not low:
            return
        second = self.ex.pick_second_model(exclude={res.model})
        if second is None:
            res.notes.append("low-confidence facts kept unconfirmed: no second model available")
            return
        b = await self.call(50, second)
        confirmed = set()
        if b.valid:
            other = {(f["entity_key"], tuple(sorted(llm_nodes.nt.numeric_tokens(f.get("content", "")))))
                     for f in b.parsed.get("facts", [])}
            for f in low:
                sig = (f["entity_key"], tuple(sorted(llm_nodes.nt.numeric_tokens(f.get("content", "")))))
                if sig in other:
                    confirmed.add(f["entity_key"])
        res.attempts.append(b)
        dropped = [f["entity_key"] for f in low if f["entity_key"] not in confirmed]
        res.parsed = {**res.parsed, "facts": [f for f in res.parsed["facts"]
                                              if f["entity_key"] not in dropped]}
        res.notes.append(f"low-confidence check by {second}: confirmed {sorted(confirmed)}, dropped {dropped}")
