"""The plan/act/observe/critique state machine.

Transition table (see `_dispatch`):

    INIT     -> PLAN     always
    PLAN     -> ACT      action non-null
    PLAN     -> DONE     planner set done=true
    PLAN     -> PLAN     parse failure, parse_retries < 2
    PLAN     -> FAILED   parse_retries >= 2
    ACT      -> OBSERVE  always (a tool error IS an observation)
    OBSERVE  -> CRITIQUE always
    CRITIQUE -> GATE     verdict == pass
    CRITIQUE -> ACT      verdict == fail and attempt_count < MAX_ATTEMPTS
    CRITIQUE -> FAILED   verdict == fail and attempt_count >= MAX_ATTEMPTS
    GATE     -> PERSIST  not externally_visible
    GATE     -> FAILED   externally_visible (Sprint 5 stub)
    PERSIST  -> PLAN     tasks remain
    PERSIST  -> DONE     no tasks remain

PLAN -> PLAN is not in the original table but is implied by it: parse_retries
cannot reach 2 without a retry edge to accumulate on.

Also not in the table: PLAN -> FAILED when the planner's new batch would push
this run's total task count over MAX_TASKS_PER_RUN. This is the PRIMARY guard
against a runaway planner. MAX_STEPS below is a backstop, not this guard.

STEP INDEX INVARIANT
    ctx.step_index    = number of steps completed = index of the NEXT step
    checkpoint.step_index = index of the step that checkpoint records
                          = ctx.step_index as step() was entered
    payload.step_index    = checkpoint.step_index + 1

    So resuming from the newest checkpoint yields a context whose step_index is
    exactly the next index to write, and indices stay contiguous with no gaps
    and no duplicates across any number of resumes.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from argus.orchestrator import fake_llm
from argus.orchestrator.context import RunContext, utcnow_iso
from argus.orchestrator.state import (
    TASK_STATUS_DONE,
    TASK_STATUS_FAILED,
    TASK_STATUS_PENDING,
    TASK_STATUS_RUNNING,
    TERMINAL_STATES,
    State,
)
from argus.storage import db

#: Two different runaway modes get two different guards, not one global step
#: cap: a single fixed number conflates "the planner won't stop making tasks"
#: with "one task is stuck retrying", and needs re-tuning by hand every time
#: steps-per-task changes.

#: PRIMARY GUARD. Enforced at PLAN: if the planner's new batch plus the tasks
#: already completed this run would exceed this, the run fails immediately.
#: This is the guard meant to actually trip on a runaway planner.
MAX_TASKS_PER_RUN = 20

#: Steps to fully process one task on the worst case (max retries) path:
#:   1 PLAN (dequeue) + 3 x (ACT + OBSERVE + CRITIQUE) + GATE + PERSIST
#: = 1 + 9 + 1 + 1 = 12
STEPS_PER_TASK = 12

#: BACKSTOP, not the primary guard. Derived from the two constants above
#: rather than hardcoded, so it tracks them automatically instead of silently
#: going stale, with ~20% slack so it can never fire before MAX_TASKS_PER_RUN
#: does on a comparable run. If this backstop trips instead of the task guard,
#: something unanticipated is happening — investigate rather than raise it.
_BASE_MAX_STEPS = 1 + MAX_TASKS_PER_RUN * STEPS_PER_TASK  # +1 for INIT -> PLAN
MAX_STEPS = int(_BASE_MAX_STEPS * 1.2)

#: Attempts at one task before CRITIQUE gives up.
MAX_ATTEMPTS = 3

#: Consecutive planner parse failures before PLAN gives up.
MAX_PARSE_RETRIES = 2

#: A DB write to run inside the step's transaction: (sql, params).
DbOp = Tuple[str, Tuple[Any, ...]]


@dataclass
class Stubs:
    """Injection point for the model-backed calls. Sprint 1 uses the fakes."""

    plan: Callable[[RunContext], Dict[str, Any]] = fake_llm.fake_plan
    act: Callable[[RunContext], Dict[str, Any]] = fake_llm.fake_act
    observe: Callable[[RunContext], Dict[str, Any]] = fake_llm.fake_observe
    critique: Callable[[RunContext], Dict[str, Any]] = fake_llm.fake_critique
    write: Callable[[RunContext], Dict[str, Any]] = fake_llm.fake_write


DEFAULT_STUBS = Stubs()


@contextmanager
def _conn_for(
    conn: Optional[sqlite3.Connection], db_path: Optional[db.PathLike]
) -> Iterator[sqlite3.Connection]:
    if conn is not None:
        yield conn
    else:
        with db.connect(db_path if db_path is not None else db.DB_PATH) as owned:
            yield owned


# ---------------------------------------------------------------------------
# task lifecycle writes
# ---------------------------------------------------------------------------


def _terminate_task(
    ctx: RunContext, status: str, error: Optional[str] = None
) -> List[DbOp]:
    """Write the task's terminal lifecycle row.

    `attempts` is written HERE and only here — the final count, once, as the
    task leaves the loop. Migration 001c's CHECK enforces the pairing:
    terminal status <=> attempts non-null, so writing one without the other
    raises rather than silently drifting.
    """
    if ctx.current_task is None:
        return []
    return [
        (
            "UPDATE tasks SET status = ?, attempts = ?, error = ? WHERE task_id = ?",
            (status, ctx.attempt_count, error, ctx.current_task["task_id"]),
        )
    ]


# ---------------------------------------------------------------------------
# per-state handlers: each mutates ctx in memory and returns (next_state, ops)
# ---------------------------------------------------------------------------


def _on_init(ctx: RunContext, stubs: Stubs) -> Tuple[State, List[DbOp]]:
    ops: List[DbOp] = [
        (
            "INSERT INTO runs (run_id, target_id, status, started_at) VALUES (?, ?, ?, ?)",
            (ctx.run_id, ctx.target_id, "running", ctx.started_at),
        )
    ]
    return State.PLAN, ops


def _on_plan(ctx: RunContext, stubs: Stubs) -> Tuple[State, List[DbOp]]:
    ops: List[DbOp] = []

    if not ctx.task_queue:
        result = stubs.plan(ctx)

        if not result.get("parse_ok", True):
            ctx.parse_retries += 1
            if ctx.parse_retries >= MAX_PARSE_RETRIES:
                ctx.reflections.append("planner output unparseable, giving up")
                return State.FAILED, _terminate_task(
                    ctx,
                    TASK_STATUS_FAILED,
                    f"planner output unparseable after {ctx.parse_retries} retries",
                )
            ctx.reflections.append("planner output unparseable, retrying")
            return State.PLAN, ops

        ctx.parse_retries = 0

        if result.get("done") or not result.get("tasks"):
            return State.DONE, ops

        # PRIMARY GUARD against a runaway planner: task_queue is always empty
        # here (that's why the planner was called), so len(facts_staged) is
        # exactly the count already completed this run.
        already_completed = len(ctx.facts_staged)
        total_after_batch = already_completed + len(result["tasks"])
        if total_after_batch > MAX_TASKS_PER_RUN:
            ctx.reflections.append(
                f"planner requested {len(result['tasks'])} more tasks "
                f"({total_after_batch} total this run), exceeding "
                f"MAX_TASKS_PER_RUN={MAX_TASKS_PER_RUN}; failing the run"
            )
            return State.FAILED, _terminate_task(
                ctx,
                TASK_STATUS_FAILED,
                f"planner exceeded MAX_TASKS_PER_RUN={MAX_TASKS_PER_RUN}",
            )

        # Global index across ALL planner calls this run, not per-batch: a
        # per-batch offset (enumerate() restarting at 0 each call) collides
        # task_id/task_index with an earlier batch's completed tasks the
        # moment the planner is called more than once in one run.
        for local_offset, task in enumerate(result["tasks"]):
            global_index = already_completed + local_offset
            ctx.task_queue.append(
                {
                    "task_id": f"{ctx.run_id}:task-{global_index}",
                    "task_index": global_index,
                    "description": task["description"],
                    "tool": task.get("tool"),
                    "action": task["action"],
                }
            )

    task = ctx.task_queue.pop(0)
    ctx.current_task = task
    ctx.attempt_count = 0
    ctx.last_action = task["action"]
    ctx.last_observation = None

    ops.append(
        (
            "INSERT INTO tasks (task_id, run_id, task_index, description, tool, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                task["task_id"],
                ctx.run_id,
                task["task_index"],
                task["description"],
                task.get("tool"),
                TASK_STATUS_PENDING,
            ),
        )
    )

    if ctx.last_action is None:
        return State.DONE, ops
    return State.ACT, ops


def _on_act(ctx: RunContext, stubs: Stubs) -> Tuple[State, List[DbOp]]:
    ctx.attempt_count += 1
    ctx.last_action = {**(ctx.last_action or {}), **stubs.act(ctx)}

    ops: List[DbOp] = []
    if ctx.current_task is not None:
        # Lifecycle moves to `running`; attempts stays NULL until termination.
        ops.append(
            (
                "UPDATE tasks SET status = ? WHERE task_id = ?",
                (TASK_STATUS_RUNNING, ctx.current_task["task_id"]),
            )
        )
    return State.OBSERVE, ops


def _on_observe(ctx: RunContext, stubs: Stubs) -> Tuple[State, List[DbOp]]:
    # A tool error is an observation, not an exception: OBSERVE always advances.
    ctx.last_observation = stubs.observe(ctx)
    return State.CRITIQUE, []


def _on_critique(ctx: RunContext, stubs: Stubs) -> Tuple[State, List[DbOp]]:
    verdict = stubs.critique(ctx)

    if verdict.get("verdict") == "pass":
        return State.GATE, []

    if verdict.get("reflection"):
        ctx.reflections.append(verdict["reflection"])

    if ctx.attempt_count < MAX_ATTEMPTS:
        return State.ACT, []

    return State.FAILED, _terminate_task(
        ctx,
        TASK_STATUS_FAILED,
        f"critique failed after {ctx.attempt_count} attempts",
    )


def _on_gate(ctx: RunContext, stubs: Stubs) -> Tuple[State, List[DbOp]]:
    action = ctx.last_action or {}
    if action.get("externally_visible"):
        # Sprint 5 replaces this with AWAIT_APPROVAL + a row in `approvals`.
        # The branch exists now so it is reachable and loggable rather than
        # being retrofitted into a loop that never had a place for it.
        ctx.reflections.append(
            "action is externally visible; blocked pending approval "
            "(Sprint 5 stub: failing instead of awaiting)"
        )
        return State.FAILED, _terminate_task(
            ctx,
            TASK_STATUS_FAILED,
            "externally visible action requires approval (Sprint 5)",
        )
    return State.PERSIST, []


def _on_persist(ctx: RunContext, stubs: Stubs) -> Tuple[State, List[DbOp]]:
    ctx.facts_staged.append(stubs.write(ctx))
    ops = _terminate_task(ctx, TASK_STATUS_DONE)

    # Cleared on BOTH branches: after PERSIST no task is in flight. Leaving it
    # set lets a later _terminate_task (the circuit breaker, most easily) flip
    # an already-completed task from done back to failed.
    ctx.current_task = None

    # SINGLE-SHOT PLANNER, not true ReAct — confirmed empirically (see
    # notes/sprint-1.md): a stub planner returning done=True on a hypothetical
    # second call was never invoked a second time. This is why: PERSIST -> PLAN
    # only fires when task_queue is non-empty (the condition below), and PLAN's
    # "queue is empty, ask the planner" branch only fires when it's empty.
    # Those two conditions are mutually exclusive, so a second planner call is
    # structurally unreachable. Sprint 1 is plan-and-execute, not interleaved
    # reasoning and acting.
    #
    # Deliberate for Sprint 1: the stub tools produce nothing worth re-planning
    # against. Revisit in Sprint 4, when real tool failures need an adaptive
    # response — changing this means routing on something other than
    # local-queue-length (e.g. an explicit "replan" signal from CRITIQUE/OBSERVE
    # instead of queue emptiness deciding PLAN vs DONE).
    if ctx.task_queue:
        return State.PLAN, ops
    return State.DONE, ops


_HANDLERS = {
    State.INIT: _on_init,
    State.PLAN: _on_plan,
    State.ACT: _on_act,
    State.OBSERVE: _on_observe,
    State.CRITIQUE: _on_critique,
    State.GATE: _on_gate,
    State.PERSIST: _on_persist,
}


def _dispatch(ctx: RunContext, stubs: Stubs) -> Tuple[State, List[DbOp]]:
    handler = _HANDLERS.get(ctx.state)
    if handler is None:
        raise ValueError(f"no handler for state {ctx.state!r}")
    return handler(ctx, stubs)


# ---------------------------------------------------------------------------
# step
# ---------------------------------------------------------------------------


def step(
    ctx: RunContext,
    conn: Optional[sqlite3.Connection] = None,
    db_path: Optional[db.PathLike] = None,
    stubs: Stubs = DEFAULT_STUBS,
) -> RunContext:
    """Advance one transition and checkpoint it, atomically.

    The state change on the context and the checkpoint row recording it are
    written in ONE transaction. A crash between them is therefore impossible:
    either both land or neither does, and a resume replays from the last
    fully-completed step.
    """
    if ctx.state in TERMINAL_STATES:
        raise ValueError(f"cannot step a terminated run (state={ctx.state.value})")

    entry_index = ctx.step_index
    task_before = ctx.current_task["task_id"] if ctx.current_task else None

    if entry_index >= MAX_STEPS:
        # BACKSTOP only — MAX_TASKS_PER_RUN in _on_plan is meant to catch a
        # runaway planner first. Reaching here means something unanticipated
        # is happening. Fires AT MAX_STEPS, so MAX_STEPS work steps (indices
        # 0..MAX_STEPS-1) actually happen and this terminal row lands at index
        # MAX_STEPS: the FAILED checkpoint records why the run stopped, it is
        # not itself a unit of work.
        ctx.reflections.append(
            f"step backstop: reached {MAX_STEPS} steps without tripping "
            f"MAX_TASKS_PER_RUN={MAX_TASKS_PER_RUN}, forcing FAILED"
        )
        next_state, ops = State.FAILED, _terminate_task(
            ctx, TASK_STATUS_FAILED, f"step backstop at {MAX_STEPS} steps"
        )
    else:
        next_state, ops = _dispatch(ctx, stubs)

    ctx.state = next_state
    ctx.step_index = entry_index + 1
    ctx.updated_at = utcnow_iso()

    # Fall back to the task the step started on, so the PERSIST checkpoint still
    # names the task it finished rather than losing it to the clear above.
    task_after = ctx.current_task["task_id"] if ctx.current_task else None
    task_id = task_after or task_before

    with _conn_for(conn, db_path) as active:
        with db.atomic(active):
            for sql, params in ops:
                active.execute(sql, params)

            active.execute(
                "INSERT INTO checkpoints "
                "(checkpoint_id, run_id, task_id, step_index, state, payload) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    ctx.run_id,
                    task_id,
                    entry_index,
                    next_state.value,
                    ctx.to_json(),
                ),
            )

            if next_state in TERMINAL_STATES:
                active.execute(
                    "UPDATE runs SET status = ?, finished_at = ? WHERE run_id = ?",
                    (
                        "done" if next_state is State.DONE else "failed",
                        ctx.updated_at,
                        ctx.run_id,
                    ),
                )

    return ctx


def run_to_completion(
    ctx: RunContext,
    conn: Optional[sqlite3.Connection] = None,
    db_path: Optional[db.PathLike] = None,
    stubs: Stubs = DEFAULT_STUBS,
    max_steps: Optional[int] = None,
) -> RunContext:
    """Step until terminal, or until `max_steps` steps have been taken here.

    `max_steps` is the test/CLI kill switch — it stops driving the loop. It is
    NOT the circuit breaker; that is MAX_STEPS inside step() and it forces
    FAILED. Stopping here leaves the run resumable.
    """
    taken = 0
    with _conn_for(conn, db_path) as active:
        while ctx.state not in TERMINAL_STATES:
            if max_steps is not None and taken >= max_steps:
                break
            step(ctx, conn=active, stubs=stubs)
            taken += 1
    return ctx


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------


def load_latest_checkpoint(
    run_id: str,
    conn: Optional[sqlite3.Connection] = None,
    db_path: Optional[db.PathLike] = None,
) -> Optional[RunContext]:
    """Rebuild the context from the newest checkpoint for `run_id`.

    ORDER BY step_index DESC is the whole resume guarantee — taking any other
    row replays already-completed work and duplicates step indices.
    """
    with _conn_for(conn, db_path) as active:
        row = active.execute(
            "SELECT payload FROM checkpoints WHERE run_id = ? "
            "ORDER BY step_index DESC LIMIT 1",
            (run_id,),
        ).fetchone()

    if row is None:
        return None
    return RunContext.from_json(row["payload"])


def new_run(target_id: str, goal: str = "", run_id: Optional[str] = None) -> RunContext:
    return RunContext(
        run_id=run_id or f"run-{uuid.uuid4().hex[:12]}",
        target_id=target_id,
        goal=goal or f"check {target_id} for changes",
    )
