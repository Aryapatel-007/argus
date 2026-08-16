"""Loop phases for the plan/act/observe/critique state machine.

TWO DIFFERENT AXES — DO NOT UNIFY THEM
======================================
The State enum below describes WHERE INSIDE PROCESSING A TASK the loop is:
PLAN, ACT, OBSERVE, CRITIQUE, GATE, PERSIST. A single task passes through these
phases many times (act, critique, retry, critique again...).

`tasks.status` in the database is a DIFFERENT axis: pending / running / done /
failed. That is the task's OWN LIFECYCLE — its terminal state, not its position
inside the loop.

They look similar because both contain a notion of "done" and "failed", and it
is tempting to merge them into one enum. That would be wrong. A task can be in
phase CRITIQUE (loop position) while its lifecycle status is still `running`.
DONE/FAILED here mean "the run's loop has terminated"; done/failed there mean
"this task reached its terminal state".

A future refactor that unifies these is a bug, not a cleanup.
"""

from __future__ import annotations

from enum import Enum


class State(str, Enum):
    """A phase within processing a task. See the module docstring."""

    INIT = "INIT"
    PLAN = "PLAN"
    ACT = "ACT"
    OBSERVE = "OBSERVE"
    CRITIQUE = "CRITIQUE"
    GATE = "GATE"
    # AWAIT_APPROVAL slots in HERE (Sprint 5): GATE -> AWAIT_APPROVAL guarded on
    # externally_visible, blocking on a row in the `approvals` table. Until then
    # GATE -> FAILED is the stub for the externally-visible branch.
    PERSIST = "PERSIST"
    DONE = "DONE"
    FAILED = "FAILED"


#: The loop stops when it reaches one of these.
TERMINAL_STATES = frozenset({State.DONE, State.FAILED})


#: Lifecycle values for `tasks.status`. Deliberately a plain tuple of strings and
#: NOT a State member — see the module docstring. Mirrors the CHECK constraint
#: added by migration 001c.
TASK_STATUS_PENDING = "pending"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_DONE = "done"
TASK_STATUS_FAILED = "failed"
