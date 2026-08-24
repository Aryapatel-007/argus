"""Embedding generation for the memory server.

One job: turn text into a 768-float vector, log the call, and hand back
something `sqlite-vec` can store. No SQL beyond the `llm_calls` row, no tool
definitions, no server plumbing — those are the memory server's, and this
module is deliberately usable without it.

Why this lives in the memory server's process rather than the orchestrator's:
the tool surface has to stay semantic (`query: str`) because the caller is the
planner LLM, write and query embeddings must come from the same model by
construction, and 768 floats as JSON is ~12KB per call over stdio. Ollama is a
shared server process, so there is no VRAM or load cost to the client living
here. See design-decisions.md.

Async because MCP v2 tools are always `async def` (CLAUDE.md), and a blocking
call inside one would occupy the event loop for the ~0.9s an embed takes.

Every call obeys the same conventions as the chat path:
  - model tag from config, never hardcoded
  - a hard 120s timeout
  - exactly one `llm_calls` row per call, INCLUDING calls that time out or fail
"""

from __future__ import annotations

import sqlite3
from typing import Any, List, Optional, Sequence, Tuple, Union

import httpx
import sqlite_vec

from argus import config
from argus.ollama_client import TIMEOUT_SECONDS, get_async_client, log_call

#: Fixed by nomic-embed-text and by `facts_vec`'s FLOAT[768] column in
#: migration 002. NOT a tunable: changing the embedding model means a new
#: migration and a full re-embed, not an edit here. `test_embeddings.py` reads
#: the width back out of the live schema and asserts it matches this constant,
#: so the two cannot drift apart silently.
EMBEDDING_DIMENSIONS = 768

#: float32, 4 bytes each. What sqlite_vec.serialize_float32 must produce.
BYTES_PER_FLOAT = 4
SERIALIZED_BYTES = EMBEDDING_DIMENSIONS * BYTES_PER_FLOAT

#: Thinking is not a concept for an embedding call, but llm_calls.think_mode is
#: NOT NULL. 'n/a' is honest; 'False' would imply a knob that was turned off.
THINK_MODE = "n/a"

ROLE = "embedder"

#: Matches the chat path, and for the same reason: it is what keeps the model
#: resident so a per-fact embed skips the cold load.
KEEP_ALIVE = "30m"


class EmbeddingError(Exception):
    """Base for every failure this module raises deliberately."""


class EmbeddingDimensionError(EmbeddingError):
    """A vector came back the wrong width.

    Named rather than a bare AssertionError for two reasons: assertions vanish
    under `python -O`, and this is not an internal invariant — it is the
    boundary check that catches the day someone points `models.embed` at a
    model with a different output width. The vector would otherwise be rejected
    much later, by sqlite-vec, with a message about a column rather than a
    model.
    """

    def __init__(self, expected: int, actual: int, model_tag: str, index: int = 0):
        self.expected = expected
        self.actual = actual
        self.model_tag = model_tag
        self.index = index
        super().__init__(
            f"embedding {index} from model {model_tag!r} has {actual} dimensions, "
            f"expected {expected}. facts_vec is declared FLOAT[{expected}] "
            f"(migration 002) and cannot store this. If the embedding model "
            f"changed, that needs a migration and a full re-embed, not a "
            f"widened check."
        )


class EmbeddingCallError(EmbeddingError):
    """The Ollama call failed or timed out. The llm_calls row is still written."""

    def __init__(self, message: str, *, timed_out: bool = False):
        self.timed_out = timed_out
        super().__init__(message)


class EmbeddingCountError(EmbeddingError):
    """Ollama returned a different number of vectors than inputs given.

    Guards the case where a single-string call comes back with an empty
    `embeddings` list: without this, indexing [0] to unwrap the result would
    raise IndexError from inside the happy path, several frames away from the
    thing that actually went wrong.
    """

    def __init__(self, expected: int, actual: int, model_tag: str):
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"model {model_tag!r} returned {actual} vector(s) for {expected} input(s)"
        )


def model_tag() -> str:
    """The embedding model, from config. Never the chat model, never a literal."""
    return config.model_for_role(ROLE)


def _validate(vectors: List[List[float]], tag: str) -> List[List[float]]:
    for index, vector in enumerate(vectors):
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise EmbeddingDimensionError(
                EMBEDDING_DIMENSIONS, len(vector), tag, index
            )
    return vectors


def serialize(vector: Sequence[float]) -> bytes:
    """Pack one vector into the bytes sqlite-vec stores.

    Wrapped so no caller ever touches struct packing directly, and so the
    width check happens on the way in rather than surfacing later as a
    sqlite-vec "Dimension mismatch" during an INSERT.
    """
    if len(vector) != EMBEDDING_DIMENSIONS:
        raise EmbeddingDimensionError(EMBEDDING_DIMENSIONS, len(vector), model_tag())

    blob = sqlite_vec.serialize_float32(list(vector))
    if len(blob) != SERIALIZED_BYTES:
        raise EmbeddingDimensionError(
            SERIALIZED_BYTES, len(blob), model_tag()
        )
    return blob


async def _invoke(tag: str, inputs: Union[str, List[str]]) -> Any:
    """One Ollama embed call. The only place this module touches the network."""
    client = get_async_client(config.ollama_options().get("host"), TIMEOUT_SECONDS)
    return await client.embed(model=tag, input=inputs, keep_alive=KEEP_ALIVE)


async def embed(
    text: Union[str, Sequence[str]],
    *,
    run_id: str,
    conn: Optional[sqlite3.Connection] = None,
    task_id: Optional[str] = None,
    step_index: Optional[int] = None,
) -> Union[List[float], List[List[float]]]:
    """Embed one string or a list of them.

    Returns a single 768-float vector for a string input, or a list of them for
    a sequence input — the shape mirrors the input, so a caller with one fact
    never has to unwrap a one-element list.

    `run_id` is passed in, never discovered: the memory server gets it from
    --run-id at spawn, and a module that went looking for "the current run"
    would be guessing on behalf of a process that already knows.

    Writes exactly one llm_calls row through the shared `log_call` path,
    including on timeout — a call that cost 120s and failed is data, and
    skipping the insert is how that cost becomes invisible.
    """
    single = isinstance(text, str)
    inputs: List[str] = [text] if single else list(text)
    tag = model_tag()

    response, timed_out, error = await _attempt(tag, inputs)

    vectors: Optional[List[List[float]]] = None
    shape_error: Optional[EmbeddingError] = None
    if response is not None:
        raw = [list(v) for v in (getattr(response, "embeddings", None) or [])]
        try:
            if len(raw) != len(inputs):
                raise EmbeddingCountError(len(inputs), len(raw), tag)
            vectors = _validate(raw, tag)
        except EmbeddingError as exc:
            shape_error = exc
            error = str(exc)

    # Logged before anything is raised, so a dimension mismatch still leaves a
    # record of the call that produced it.
    log_call(
        conn,
        run_id=run_id,
        task_id=task_id,
        step_index=step_index,
        role=ROLE,
        model_tag=tag,
        think_mode=THINK_MODE,
        prompt_file=None,
        response=response,
        ok=vectors is not None,
        timed_out=timed_out,
        error=error,
        # The "raw response" is the vector itself; it belongs in facts_vec, not
        # duplicated as text here.
        raw_response=None,
    )

    if shape_error is not None:
        raise shape_error
    if vectors is None:
        raise EmbeddingCallError(error or "embedding call failed", timed_out=timed_out)

    return vectors[0] if single else vectors


async def _attempt(
    tag: str, inputs: List[str]
) -> Tuple[Any, bool, Optional[str]]:
    """Returns (response, timed_out, error). Never raises — the caller logs first.

    Mirrors llm._attempt, including why TimeoutException is caught separately:
    ConnectTimeout/ReadTimeout are not httpx.ConnectError, so ollama's internal
    handler does not swallow them and they arrive here intact.
    """
    try:
        return await _invoke(tag, inputs), False, None
    except httpx.TimeoutException as exc:
        return None, True, f"timeout after {TIMEOUT_SECONDS}s: {exc!r}"
    except ConnectionError as exc:
        return None, False, f"ollama unreachable: {exc}"
    except Exception as exc:  # noqa: BLE001 - a failed call must still be logged
        return None, False, f"{type(exc).__name__}: {exc}"
