"""Model calls for the graph engine. Async, one seam, every call logged.

CLAUDE.md conventions, all enforced here and nowhere else in the graph:
  - `think=False` is a TOP-LEVEL chat field on every call, every model.
  - a hard 120s cap: the client's httpx timeout AND an asyncio.wait_for
    wall-clock cap around the whole call (audit L13: the httpx timeout is
    per-phase, not total).
  - exactly one llm_calls row per call, including timeouts and schema
    failures, through `ollama_client.log_call` (the single insert path).
  - model tags arrive from the router; nothing here picks a model.

A logging failure never kills the call that already happened (audit M6): it
is reported on stderr and the result is still returned.

Tests and the eval harness swap `OllamaBackend` for a fake with the same
`chat` signature. That is the one seam (argus-agentic-testing skill).
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Type

import httpx
from pydantic import BaseModel, ValidationError

from argus import config
from argus.ollama_client import TIMEOUT_SECONDS, get_async_client, log_call, tok_s

THINK = False
NS_PER_S = 1_000_000_000


@dataclass
class LogContext:
    run_id: str
    role: str
    conn: Optional[sqlite3.Connection] = None
    graph_id: Optional[str] = None
    node_id: Optional[str] = None
    strategy: Optional[str] = None
    sample_index: Optional[int] = None
    decision_id: Optional[str] = None
    prompt_file: Optional[str] = None


@dataclass
class CallResult:
    model_tag: str
    parsed: Optional[Dict[str, Any]]
    raw: Optional[str]
    error: Optional[str]
    timed_out: bool = False
    eval_count: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    tok_s: Optional[float] = None
    load_s: Optional[float] = None
    wall_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.parsed is not None


class ModelBackend(Protocol):
    async def chat(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        schema: Type[BaseModel],
        log: LogContext,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> CallResult: ...

    async def resident_models(self) -> List[str]: ...


def _log(log: LogContext, model: str, response: Any, ok: bool, timed_out: bool,
         error: Optional[str], raw: Optional[str]) -> None:
    try:
        log_call(
            log.conn, run_id=log.run_id, task_id=None, step_index=None, role=log.role,
            model_tag=model, think_mode=str(THINK), prompt_file=log.prompt_file,
            response=response, ok=ok, timed_out=timed_out, error=error, raw_response=raw,
            graph_id=log.graph_id, node_id=log.node_id, strategy=log.strategy,
            sample_index=log.sample_index, decision_id=log.decision_id,
        )
    except Exception as exc:  # noqa: BLE001 - observability must not kill a paid call
        print(f"[argus.graph.llm] llm_calls insert failed: {exc!r}", file=sys.stderr, flush=True)


def parse(schema: Type[BaseModel], raw: Optional[str]) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        return schema.model_validate_json(raw or "").model_dump(), None
    except ValidationError as exc:
        return None, f"schema validation failed: {exc}"


class OllamaBackend:
    """The real thing. One AsyncClient per event loop (ollama_client cache)."""

    def __init__(self, host: Optional[str] = None, num_ctx: Optional[int] = None,
                 keep_alive: Optional[str] = None):
        opts = config.ollama_options()
        self.host = host or opts.get("host")
        self.num_ctx = num_ctx or opts.get("num_ctx")
        self.keep_alive = keep_alive or opts.get("keep_alive", "30m")

    async def chat(self, *, model, messages, schema, log, temperature=0.0, seed=None) -> CallResult:
        options: Dict[str, Any] = {"temperature": temperature}
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        if seed is not None:
            options["seed"] = int(seed)
        client = get_async_client(self.host, TIMEOUT_SECONDS)
        start = time.perf_counter()
        response, timed_out, error, raw = None, False, None, None
        try:
            response = await asyncio.wait_for(
                client.chat(
                    model=model,
                    messages=messages,
                    think=THINK,  # top-level, explicit, every call
                    keep_alive=self.keep_alive,
                    format=schema.model_json_schema(),
                    options=options,
                ),
                timeout=TIMEOUT_SECONDS,
            )
            raw = response.message.content if response.message else None
        except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
            timed_out, error = True, f"timeout after {TIMEOUT_SECONDS}s: {exc!r}"
        except Exception as exc:  # noqa: BLE001 - every failure is logged and returned
            error = f"{type(exc).__name__}: {exc}"
        wall = time.perf_counter() - start

        parsed = None
        if response is not None:
            parsed, error = parse(schema, raw)
        _log(log, model, response, parsed is not None, timed_out, error, raw)
        eval_count = getattr(response, "eval_count", None)
        load = getattr(response, "load_duration", None)
        return CallResult(
            model_tag=model, parsed=parsed, raw=raw, error=error, timed_out=timed_out,
            eval_count=eval_count, prompt_eval_count=getattr(response, "prompt_eval_count", None),
            tok_s=tok_s(eval_count, getattr(response, "eval_duration", None)),
            load_s=(load / NS_PER_S) if load else None, wall_s=wall,
        )

    async def resident_models(self) -> List[str]:
        client = get_async_client(self.host, TIMEOUT_SECONDS)
        try:
            ps = await asyncio.wait_for(client.ps(), timeout=10)
        except Exception:  # noqa: BLE001 - residency is a hint, never fatal
            return []
        return [config.normalize_tag(m.model) for m in (ps.models or [])]
