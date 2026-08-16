"""RunContext: the full state of a run, checkpointed as JSON after every step.

This object is the resume unit. Everything needed to continue a run after a
crash must live here, because `checkpoints.payload` holds exactly this and
nothing else. Anything kept only in memory (a module-level counter, an open
handle) is lost on resume and will make a resumed run diverge from an
uninterrupted one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from argus.orchestrator.state import State


def utcnow_iso() -> str:
    """Timestamp in the same shape as the SQL defaults in migration 001."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@dataclass
class RunContext:
    run_id: str
    target_id: str
    goal: str

    state: State = State.INIT
    step_index: int = 0

    task_queue: List[Dict[str, Any]] = field(default_factory=list)
    current_task: Optional[Dict[str, Any]] = None

    #: Attempts at the CURRENT task. Reset when a new task is dequeued. This is
    #: the live counter during a run; `tasks.attempts` is only written once, at
    #: task termination. Two tables must never both claim live truth.
    attempt_count: int = 0

    parse_retries: int = 0

    last_action: Optional[Dict[str, Any]] = None
    last_observation: Optional[Dict[str, Any]] = None

    reflections: List[str] = field(default_factory=list)
    facts_staged: List[Dict[str, Any]] = field(default_factory=list)

    started_at: str = field(default_factory=utcnow_iso)
    updated_at: str = field(default_factory=utcnow_iso)

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Serialize the enum's value explicitly. State subclasses str, so
        # json.dumps would happen to emit "PLAN" anyway, but str(State.PLAN)
        # emits "State.PLAN" — too easy to reintroduce that bug by accident.
        data["state"] = self.state.value
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunContext":
        data = dict(data)
        data["state"] = State(data["state"])
        return cls(**data)

    @classmethod
    def from_json(cls, payload: str) -> "RunContext":
        return cls.from_dict(json.loads(payload))
