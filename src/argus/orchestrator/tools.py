"""Three stub tools for ACT. No real MCP this sprint.

They cover the three result shapes the critic has to tell apart: a usable
result, a hard failure, and the quiet one — a call that succeeds but returns
nothing. The empty case is the interesting one, because a critic that reads
`ok: true` and stops will pass it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict


class ToolError(RuntimeError):
    """Raised by a tool that fails. ACT catches it — a tool error is an
    observation, never an exception that escapes the loop."""


def stub_fetch_ok(args: Dict[str, Any]) -> Dict[str, Any]:
    """Always succeeds with usable content."""
    return {
        "content": (
            "Master of Artificial Intelligence — intake dates: February and "
            "July 2027. Duration: 2 years full-time."
        ),
        "args_echo": args,
    }


def stub_fetch_fails(args: Dict[str, Any]) -> Dict[str, Any]:
    """Always fails."""
    raise ToolError("upstream returned HTTP 503 after 3 attempts")


def stub_fetch_empty(args: Dict[str, Any]) -> Dict[str, Any]:
    """Succeeds, but with nothing in it."""
    return {"content": "", "args_echo": args}


REGISTRY: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "stub_fetch_ok": stub_fetch_ok,
    "stub_fetch_fails": stub_fetch_fails,
    "stub_fetch_empty": stub_fetch_empty,
}

#: Tools whose execution leaves the machine. Declared by the TOOL, statically,
#: never inferred from the planner. Audit C1: the planner's
#: `externally_visible` flag alone let an external tool marked false run with
#: no approval. A real outbound tool must be listed here when registered.
SIDE_EFFECT_TOOLS: frozenset = frozenset()


def is_external(tool_name: Any, planner_flag: Any) -> bool:
    """External if the tool says so OR the planner says so. Either is enough."""
    if not isinstance(tool_name, str):
        # A malformed tool name cannot be looked up; fail closed (review L-2).
        return True
    return bool(planner_flag) or tool_name in SIDE_EFFECT_TOOLS


#: Rendered into planner.md so the planner can only pick real tool names.
TOOL_CATALOG = """- `stub_fetch_ok(url: str)` — fetches a page and returns its text. Reliable.
- `stub_fetch_fails(url: str)` — fetches from a flaky upstream. Often errors.
- `stub_fetch_empty(url: str)` — fetches a page that is frequently blank."""


def run_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool and return an observation. Never raises.

    Both the failure and the unknown-tool cases come back as observations with
    `ok: false`, because ACT -> OBSERVE is unconditional: the loop must always
    get something for the critic to judge.
    """
    tool = REGISTRY.get(tool_name) if isinstance(tool_name, str) else None
    if tool is None:
        return {
            "ok": False,
            "tool": tool_name,
            "content": "",
            "error": f"unknown tool {tool_name!r}; available: {sorted(REGISTRY)}",
        }

    try:
        result = tool(args or {})
    except ToolError as exc:
        return {"ok": False, "tool": tool_name, "content": "", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - "never raises" is the contract (audit L3)
        return {
            "ok": False,
            "tool": tool_name,
            "content": "",
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "ok": True,
        "tool": tool_name,
        "content": result.get("content", ""),
        "error": None,
    }
