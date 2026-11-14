"""Real Ollama calls: schema-validated, timeout-bounded, fully logged.

Every call obeys the CLAUDE.md conventions:
  - `think=False` passed EXPLICITLY as a TOP-LEVEL field, never inside options,
    and never left to the default. Thinking is binary on qwen3.5:9b and an
    unconstrained call has been measured at 1583s returning empty content.
  - a hard 120s timeout, and tok/s logged for every call
  - model tags come from config, never from source
  - `num_predict` is NEVER used as a thinking cap (it truncates mid-reasoning
    and returns no answer)

Every call writes exactly one `llm_calls` row per attempt — including calls
that time out or fail validation. The row is written OUTSIDE the loop's step
transaction (see loop.step) so that a step which later fails still leaves
behind the record of the call that really happened and really cost 120s.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from string import Template
from typing import Any, Dict, Literal, Optional, Tuple, Type

import httpx
from pydantic import BaseModel, Field, ValidationError

from argus import config
from argus.ollama_client import (  # noqa: F401 - re-exported, see module docstring
    TIMEOUT_SECONDS,
    close_async_clients,
    get_async_client,
    get_client as _get_client,
    log_call,
    reset_async_clients,
)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

#: Top-level chat field, not an options key. See module docstring.
THINK = False
KEEP_ALIVE = "30m"
TEMPERATURE = 0

#: Attempts per call: the first, plus one retry that feeds the validation error
#: back to the model.
MAX_SCHEMA_ATTEMPTS = 2

# Sync client cache, get_async_client, log_call, TIMEOUT_SECONDS and the
# async-client lifecycle helpers now live in argus.ollama_client, shared with
# argus.embeddings. Imported and re-exported above under their original names
# (get_client as _get_client) so every existing caller and test in this
# module and its test suite keeps working unmodified.


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------


class Action(BaseModel):
    tool_name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    externally_visible: bool


class PlannerOutput(BaseModel):
    thought: str
    # A list, not a single action: the planner is single-shot per run (see the
    # SINGLE-SHOT PLANNER note in loop._on_persist), so the whole plan has to
    # arrive in one response.
    tasks: list[Action] = Field(default_factory=list)
    done: bool


class CriticOutput(BaseModel):
    # Literal, not str: it lands in the emitted JSON schema as an enum, so the
    # model is told the only two legal values, and anything else fails
    # validation and earns the retry instead of silently becoming a "fail".
    verdict: Literal["pass", "fail"]
    confidence: float
    reflection: str


ROLE_SCHEMAS: Dict[str, Type[BaseModel]] = {
    "planner": PlannerOutput,
    "critic": CriticOutput,
    # "writer" lands in Sprint 2 with the facts table; fake_write stands in.
}


# ---------------------------------------------------------------------------
# prompt rendering
# ---------------------------------------------------------------------------


def render_prompt(prompt_file: str, context_vars: Dict[str, Any]) -> str:
    """Render a template from prompts/.

    string.Template ($var), not str.format: prompts contain JSON braces and
    format() would treat them as fields.
    """
    text = (PROMPTS_DIR / prompt_file).read_text(encoding="utf-8")
    return Template(text).safe_substitute(context_vars)


# ---------------------------------------------------------------------------
# the single seam over the network — tests patch Client, in argus.ollama_client
# ---------------------------------------------------------------------------


def _invoke(model_tag: str, messages: list, schema: Dict[str, Any]) -> Any:
    """One Ollama chat call. The only place that touches the network."""
    options = {"temperature": TEMPERATURE}
    num_ctx = config.ollama_options().get("num_ctx")
    if num_ctx:
        options["num_ctx"] = num_ctx

    client = _get_client(config.ollama_options().get("host"), TIMEOUT_SECONDS)
    return client.chat(
        model=model_tag,
        messages=messages,
        think=THINK,
        keep_alive=KEEP_ALIVE,
        format=schema,
        options=options,
    )


# ---------------------------------------------------------------------------
# logging — log_call itself lives in argus.ollama_client, imported above
# ---------------------------------------------------------------------------


def _log_call(
    conn: Optional[sqlite3.Connection],
    ctx: Any,
    role: str,
    model_tag: str,
    prompt_file: str,
    response: Any,
    ok: bool,
    timed_out: bool,
    error: Optional[str],
    raw_response: Optional[str],
) -> None:
    """Chat-call adapter over `log_call`: unpacks the RunContext."""
    log_call(
        conn,
        run_id=ctx.run_id,
        task_id=ctx.current_task["task_id"] if ctx and ctx.current_task else None,
        step_index=ctx.step_index if ctx else None,
        role=role,
        model_tag=model_tag,
        think_mode=str(THINK),
        prompt_file=prompt_file,
        response=response,
        ok=ok,
        timed_out=timed_out,
        error=error,
        raw_response=raw_response,
    )


# ---------------------------------------------------------------------------
# call_ollama
# ---------------------------------------------------------------------------


def _attempt(
    model_tag: str, messages: list, schema: Dict[str, Any]
) -> Tuple[Any, bool, Optional[str], Optional[str]]:
    """Returns (response, timed_out, error, raw_content)."""
    try:
        response = _invoke(model_tag, messages, schema)
    except httpx.TimeoutException as exc:
        # ConnectTimeout/ReadTimeout are NOT httpx.ConnectError, so ollama's
        # internal handler does not swallow them and they arrive here intact.
        return None, True, f"timeout after {TIMEOUT_SECONDS}s: {exc!r}", None
    except ConnectionError as exc:
        return None, False, f"ollama unreachable: {exc}", None
    except Exception as exc:  # noqa: BLE001 - a failed call must still be logged
        return None, False, f"{type(exc).__name__}: {exc}", None

    raw = response.message.content if response.message else None
    return response, False, None, raw


def call_ollama(
    role: str,
    prompt_file: str,
    context_vars: Dict[str, Any],
    *,
    ctx: Any = None,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[BaseModel]:
    """Call Ollama for `role` and return the validated model, or None.

    On a schema validation failure, retries ONCE with the validation error fed
    back in. If that also fails — or the call times out — increments
    ctx.parse_retries and returns None so the loop's PLAN -> FAILED guard can
    fire. The increment lives here rather than in the loop because this is the
    layer that knows a call actually failed; the loop only checks the guard.
    """
    schema_model = ROLE_SCHEMAS.get(role)
    if schema_model is None:
        raise ValueError(f"no output schema registered for role {role!r}")

    model_tag = config.model_for_role(role)
    schema = schema_model.model_json_schema()
    messages = [{"role": "user", "content": render_prompt(prompt_file, context_vars)}]

    for attempt in range(1, MAX_SCHEMA_ATTEMPTS + 1):
        response, timed_out, error, raw = _attempt(model_tag, messages, schema)
        parsed: Optional[BaseModel] = None

        if response is not None:
            try:
                parsed = schema_model.model_validate_json(raw or "")
            except ValidationError as exc:
                error = f"schema validation failed: {exc}"

        _log_call(
            conn,
            ctx,
            role,
            model_tag,
            prompt_file,
            response,
            ok=parsed is not None,
            timed_out=timed_out,
            error=error,
            raw_response=raw,
        )

        if parsed is not None:
            return parsed

        # A timeout or an unreachable server will not be fixed by restating the
        # schema, so only a validation failure earns the retry.
        retryable = response is not None and not timed_out
        if attempt < MAX_SCHEMA_ATTEMPTS and retryable:
            messages = messages + [
                {"role": "assistant", "content": raw or ""},
                {
                    "role": "user",
                    "content": (
                        f"That response failed schema validation:\n{error}\n\n"
                        "Return ONLY a JSON object matching the schema. "
                        "No prose, no markdown fences."
                    ),
                },
            ]
            continue
        break

    if ctx is not None and role == "planner":
        # Covers timeouts too, not just parse failures: without an increment the
        # loop would re-enter PLAN forever on an unreachable server.
        # Planner only (audit M1): parse_retries guards PLAN, and a critic
        # timeout spending the planner's budget would fail a run on its first
        # bad plan. The critic's own failures are bounded by MAX_ATTEMPTS.
        ctx.parse_retries += 1
    return None


# ---------------------------------------------------------------------------
# loop adapters — these replace fake_plan / fake_critique
# ---------------------------------------------------------------------------


def _describe(action: Action) -> str:
    """Action carries no description field, so derive one for tasks.description."""
    return f"{action.tool_name}({json.dumps(action.args, sort_keys=True)})"


def real_plan(ctx: Any, conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
    from argus.orchestrator import tools  # local import keeps the module graph flat

    output = call_ollama(
        "planner",
        "planner.md",
        {
            "goal": ctx.goal,
            "target_id": ctx.target_id,
            "tool_catalog": tools.TOOL_CATALOG,
            "max_tasks": 5,
        },
        ctx=ctx,
        conn=conn,
    )

    if output is None:
        return {"parse_ok": False, "done": False, "tasks": []}

    return {
        "parse_ok": True,
        "done": output.done,
        "thought": output.thought,
        "tasks": [
            {
                "description": _describe(action),
                "tool": action.tool_name,
                "action": {
                    "tool": action.tool_name,
                    "args": action.args,
                    "externally_visible": action.externally_visible,
                },
            }
            for action in output.tasks
        ],
    }


def real_critique(ctx: Any, conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
    action = ctx.last_action or {}
    observation = ctx.last_observation or {}

    output = call_ollama(
        "critic",
        "critic.md",
        {
            # Deliberately narrow: the action and its result, no goal and no
            # history. See the note at the top of critic.md.
            "action": json.dumps(
                {"tool": action.get("tool"), "args": action.get("args", {})},
                indent=2,
                sort_keys=True,
            ),
            "result": json.dumps(observation, indent=2, sort_keys=True),
        },
        ctx=ctx,
        conn=conn,
    )

    if output is None:
        return {
            "verdict": "fail",
            "reflection": "critic call failed; retrying the action",
            "confidence": 0.0,
        }

    # No normalisation needed: the Literal already rejected anything that is
    # not exactly "pass" or "fail" at validation time.
    return {
        "verdict": output.verdict,
        "reflection": output.reflection,
        "confidence": output.confidence,
    }
