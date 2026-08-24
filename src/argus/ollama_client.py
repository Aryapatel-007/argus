"""Shared Ollama plumbing: cached clients and the single llm_calls insert path.

Neutral on purpose. `argus.orchestrator.llm` (chat: planner/critic/writer/
judge) and `argus.embeddings` (embedder, run by the memory server) both depend
on this module. Neither depends on the other — an embedding call has no
business importing orchestrator machinery to log a row, and the orchestrator
has no business knowing about embeddings. This module is the thing both of
them are allowed to know about.

Moved here unchanged from `argus.orchestrator.llm` in Sprint 2, once
`argus.embeddings` needed the client cache and the logging path and importing
them from `orchestrator.llm` was the layering pointing the wrong way.
"""

from __future__ import annotations

import asyncio
import functools
import sqlite3
import uuid
import weakref
from typing import Any, Dict, Optional, Tuple

from ollama import AsyncClient, Client

#: Hard timeout on every Ollama call, every role. CLAUDE.md convention.
TIMEOUT_SECONDS = 120

NS_PER_S = 1_000_000_000


# ---------------------------------------------------------------------------
# sync client cache
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=4)
def get_client(host: Optional[str], timeout: int) -> Client:
    """One client per (host, timeout), reused across calls.

    Cached because each ollama.Client wraps its own httpx.Client and therefore
    its own connection pool. Building one per call leaks a pool per call and
    throws away keep-alive, which matters here: keep_alive="30m" holds the
    model resident precisely so consecutive calls skip the 6s cold load.
    """
    return Client(host=host, timeout=timeout)


# ---------------------------------------------------------------------------
# async client cache
# ---------------------------------------------------------------------------

#: One AsyncClient per (event loop, host, timeout). Keyed weakly on the loop so
#: an entry disappears when its loop is collected — see get_async_client.
_ASYNC_CLIENTS: "weakref.WeakKeyDictionary[Any, Dict[Tuple[Optional[str], int], AsyncClient]]" = (
    weakref.WeakKeyDictionary()
)


def get_async_client(host: Optional[str], timeout: int) -> AsyncClient:
    """Async twin of `get_client`, cached for exactly the same reason.

    Added in Sprint 2 for the embedder — there was no AsyncClient before this.
    A sync and an async client cannot share a connection pool (httpx.Client and
    httpx.AsyncClient are different objects), so this is the same one-pool fix
    applied to the async side, not a second pool for the same purpose.

    Cached, because building one per call leaks a pool per call and throws away
    keep-alive — and keep-alive is what keeps nomic-embed-text resident so a
    per-fact embed costs ~0.9s instead of a reload.

    Keyed on the RUNNING EVENT LOOP as well as (host, timeout), which a plain
    lru_cache is not. An httpx.AsyncClient binds its connection pool to the
    loop that created it, so a client cached under one loop and reused under
    another fails with "RuntimeError: Event loop is closed". The memory server
    runs a single loop for its whole life and would never have hit this — the
    bug only surfaced under the test suite, which runs one loop per test.

    WeakKeyDictionary rather than the loop's id(): a dead loop's entry is
    collected with it, so a new loop allocated at the same address cannot
    inherit a stale pool.
    """
    loop = asyncio.get_running_loop()
    per_loop = _ASYNC_CLIENTS.setdefault(loop, {})
    key = (host, timeout)
    if key not in per_loop:
        per_loop[key] = AsyncClient(host=host, timeout=timeout)
    return per_loop[key]


def reset_async_clients() -> None:
    """Drop every cached AsyncClient. For tests; mirrors lru_cache.cache_clear."""
    _ASYNC_CLIENTS.clear()


async def close_async_clients() -> None:
    """Close and forget this loop's cached clients.

    Callers should invoke this on shutdown so the connection pool is released
    rather than left to the garbage collector, which on Windows surfaces as
    ResourceWarning about unclosed transports. Only touches the RUNNING loop's
    clients — another loop's pool cannot be closed from here.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    for client in _ASYNC_CLIENTS.pop(loop, {}).values():
        await client.close()


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------


def tok_s(eval_count: Optional[int], eval_duration_ns: Optional[int]) -> Optional[float]:
    """Same formula as scripts/probe_thinking.py."""
    if not eval_count or not eval_duration_ns:
        return None
    return eval_count / (eval_duration_ns / NS_PER_S)


def log_call(
    conn: Optional[sqlite3.Connection],
    *,
    run_id: str,
    task_id: Optional[str],
    step_index: Optional[int],
    role: str,
    model_tag: str,
    think_mode: str,
    prompt_file: Optional[str],
    response: Any,
    ok: bool,
    timed_out: bool,
    error: Optional[str],
    raw_response: Optional[str],
) -> None:
    """Insert one llm_calls row. Called for EVERY attempt, success or not.

    THE single insert path for llm_calls. The planner, the critic and the
    memory server's embedder all write through this function, so there is one
    place that knows the column list and one place to change when it moves.

    Identifiers are passed explicitly rather than dug out of a RunContext: the
    memory server runs in a different process, has no RunContext, and receives
    only a run_id from --run-id at spawn. A ctx-shaped shim built just to
    satisfy this signature would be a second path wearing the first one's
    clothes.

    Metrics are read off `response` with getattr, so a chat response and an
    embed response both work untouched. That is what makes eval_count,
    eval_duration_ns and tok_s land as NULL for an embedding call: Ollama's
    EmbedResponse genuinely omits them, getattr returns None, and tok_s
    returns None from None inputs. Never substitute 0 — see 002's column
    semantics comment.

    Runs outside any transaction (db.connect is autocommit), so the row commits
    immediately and survives a later rollback of the step that made the call.
    """
    if conn is None:
        return

    eval_count = getattr(response, "eval_count", None)
    eval_duration = getattr(response, "eval_duration", None)

    conn.execute(
        "INSERT INTO llm_calls ("
        "call_id, run_id, task_id, step_index, role, model_tag, think_mode, "
        "prompt_file, prompt_eval_count, eval_count, eval_duration_ns, "
        "total_duration_ns, tok_s, ok, timed_out, error, raw_response"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            str(uuid.uuid4()),
            run_id,
            task_id,
            step_index,
            role,
            model_tag,
            think_mode,
            prompt_file,
            getattr(response, "prompt_eval_count", None),
            eval_count,
            eval_duration,
            getattr(response, "total_duration", None),
            tok_s(eval_count, eval_duration),
            1 if ok else 0,
            1 if timed_out else 0,
            error,
            raw_response,
        ),
    )
