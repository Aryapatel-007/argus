"""Stubbed planner / critic / writer, shaped like the real Ollama output.

Sprint 1 proves the state machine mechanics without a model in the loop. These
return hardcoded structures matching what the real calls will produce, so
swapping in Ollama later is a change of implementation, not of contract.

RESUME SAFETY: none of these keep module-level state. Every decision is derived
from the RunContext passed in, which is checkpointed. A counter kept in a module
dict here would be lost on resume and a resumed run would take a different
number of steps than an uninterrupted one — exactly the bug the resume tests
look for.
"""

from __future__ import annotations

from typing import Any, Dict

from argus.orchestrator.context import RunContext

#: fake_critique fails this many times per task before passing, so the
#: CRITIQUE -> ACT retry edge is exercised on every task.
CRITIQUE_FAILURES_PER_TASK = 2


def fake_plan(ctx: RunContext) -> Dict[str, Any]:
    """Return the initial task list, or done once the queue is exhausted.

    Only called when the queue is empty: PERSIST routes back to PLAN solely
    while tasks remain, so a re-entry with work left never reaches the planner.
    """
    if ctx.facts_staged or ctx.current_task is not None:
        # Work has already happened and the queue is empty: nothing left to do.
        return {"done": True, "parse_ok": True, "tasks": []}

    return {
        "done": False,
        "parse_ok": True,
        "tasks": [
            {
                "description": "check intake dates",
                "tool": "open_page",
                "action": {
                    "tool": "open_page",
                    "args": {"url": "https://example.invalid/intake"},
                    "externally_visible": False,
                },
            },
            {
                "description": "check entry requirements",
                "tool": "extract_structured",
                "action": {
                    "tool": "extract_structured",
                    "args": {"fields": ["entry_requirements"]},
                    "externally_visible": False,
                },
            },
        ],
    }


def fake_act(ctx: RunContext) -> Dict[str, Any]:
    """Pretend to invoke the tool named by the current action."""
    action = ctx.last_action or {}
    return {
        "tool": action.get("tool"),
        "args": action.get("args", {}),
        "attempt": ctx.attempt_count,
    }


def fake_observe(ctx: RunContext) -> Dict[str, Any]:
    """Observe the result. A tool error is an observation, never an exception."""
    action = ctx.last_action or {}
    return {
        "ok": True,
        "tool": action.get("tool"),
        "content": f"stub result for {action.get('tool')} attempt {ctx.attempt_count}",
    }


def fake_critique(ctx: RunContext) -> Dict[str, Any]:
    """Fail the first CRITIQUE_FAILURES_PER_TASK attempts, then pass.

    Derived from ctx.attempt_count (which resets per task and is checkpointed)
    rather than a per-task_id counter held in module state — see the resume-
    safety note in the module docstring.
    """
    if ctx.attempt_count <= CRITIQUE_FAILURES_PER_TASK:
        return {
            "verdict": "fail",
            "reflection": (
                f"attempt {ctx.attempt_count} was insufficient; "
                "retry with a narrower extraction"
            ),
        }
    return {"verdict": "pass", "reflection": None}


def fake_write(ctx: RunContext) -> Dict[str, Any]:
    """Shape a fact to stage at PERSIST."""
    task = ctx.current_task or {}
    observation = ctx.last_observation or {}
    return {
        "target_id": ctx.target_id,
        "task_id": task.get("task_id"),
        "summary": f"{task.get('description')}: {observation.get('content')}",
        "attempts": ctx.attempt_count,
    }
