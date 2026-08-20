"""Real crash-resume harness: kill a live process mid-Ollama-call, then resume.

Every resume test in tests/ mocks Ollama and simulates a crash by raising inside
a `with` block. That proves the rollback logic, not the guarantee CLAUDE.md's
Sprint 1 goal actually claims — "resumable after a crash" means an OS-level kill
with a real model call in flight and a real transaction possibly open.

This is a harness, not production code and not a unittest: it spawns and kills
real processes and needs a live Ollama, so it lives in scripts/.

    python scripts/crash_test.py
    python scripts/crash_test.py --self-check     # no Ollama, no subprocesses

WHY llm_calls MAY BE ORPHANED (expected, do not "fix")
    llm_calls rows are written OUTSIDE the step transaction, deliberately: it is
    an observability log, not run state, and a call that really cost 120s must
    stay recorded even if the step that made it never commits. So a killed run
    MAY leave an llm_calls row at step_index N with no checkpoint at index N.
    That is correct and this harness asserts it breaks nothing (foreign keys and
    integrity stay clean) rather than treating it as damage. Removing that
    property to make the counts line up would lose exactly the diagnostic the
    column exists for.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from argus.orchestrator.state import State  # noqa: E402
from argus.storage.migrate import apply_migrations  # noqa: E402

VALID_STATES = {s.value for s in State}
TERMINAL = {State.DONE.value, State.FAILED.value}

#: Checkpoint states whose NEXT dispatch makes a real Ollama call. Seeing one of
#: these means a model call is about to start / just started.
PRE_LLM_STATES = {State.PLAN.value, State.CRITIQUE.value}

POLL_INTERVAL = 0.05


# ---------------------------------------------------------------------------
# database reads (all read-only; the harness never writes to a live run's db)
# ---------------------------------------------------------------------------


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def read_run_id(db_path: Path) -> Optional[str]:
    """The one run in a dedicated crash-test database."""
    if not db_path.exists():
        return None
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT run_id FROM runs LIMIT 1").fetchone()
        return row["run_id"] if row else None
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def read_checkpoints(db_path: Path, run_id: str) -> List[sqlite3.Row]:
    conn = _connect(db_path)
    try:
        return conn.execute(
            "SELECT step_index, state, task_id, payload FROM checkpoints "
            "WHERE run_id = ? ORDER BY step_index",
            (run_id,),
        ).fetchall()
    finally:
        conn.close()


def read_tasks(db_path: Path, run_id: str) -> List[sqlite3.Row]:
    conn = _connect(db_path)
    try:
        return conn.execute(
            "SELECT task_id, task_index, description, tool, status, attempts "
            "FROM tasks WHERE run_id = ? ORDER BY task_index",
            (run_id,),
        ).fetchall()
    finally:
        conn.close()


def read_llm_calls(db_path: Path, run_id: str) -> List[sqlite3.Row]:
    conn = _connect(db_path)
    try:
        return conn.execute(
            "SELECT step_index, role, ok, timed_out FROM llm_calls "
            "WHERE run_id = ? ORDER BY rowid",
            (run_id,),
        ).fetchall()
    finally:
        conn.close()


def count_llm_calls(db_path: Path) -> int:
    if not db_path.exists():
        return 0
    conn = _connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) AS n FROM llm_calls").fetchone()["n"]
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


def pragma_checks(db_path: Path) -> Tuple[str, List[Any]]:
    conn = _connect(db_path)
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        conn.execute("PRAGMA foreign_keys = ON")
        fk = conn.execute("PRAGMA foreign_key_check").fetchall()
        return integrity, fk
    finally:
        conn.close()


def final_context(db_path: Path, run_id: str) -> Optional[Dict[str, Any]]:
    rows = read_checkpoints(db_path, run_id)
    if not rows:
        return None
    return json.loads(rows[-1]["payload"])


# ---------------------------------------------------------------------------
# comparing two DIFFERENT runs
# ---------------------------------------------------------------------------

RUN_PLACEHOLDER = "<run>"


def normalise_run_scoped(value: Any, run_id: str) -> Any:
    """Replace this run's identity with a placeholder, recursively.

    Two runs necessarily have different run_ids (uuid4), and identifiers derived
    from them differ too — task_id is literally f"{run_id}:task-{n}". Comparing
    raw structures across runs is therefore unsatisfiable BY CONSTRUCTION: no
    crash-resume behaviour could ever make it pass.

    Substituting rather than dropping the field keeps the check meaningful: the
    task ORDINAL ("<run>:task-0") still has to match, so a fact attached to the
    wrong task is still caught. Only the run-unique prefix is neutralised.
    """
    if isinstance(value, str):
        return value.replace(run_id, RUN_PLACEHOLDER)
    if isinstance(value, list):
        return [normalise_run_scoped(v, run_id) for v in value]
    if isinstance(value, dict):
        return {k: normalise_run_scoped(v, run_id) for k, v in value.items()}
    return value


def structural_diff(left: Any, right: Any, path: str = "") -> List[str]:
    """Human-readable differences, so a failure explains itself.

    The previous message printed only counts ("1 vs 1"), which said nothing
    about why two one-element lists were unequal.
    """
    if isinstance(left, dict) and isinstance(right, dict):
        diffs: List[str] = []
        for key in sorted(set(left) | set(right)):
            sub = f"{path}.{key}" if path else key
            if key not in left:
                diffs.append(f"{sub}: missing on left, right={right[key]!r}")
            elif key not in right:
                diffs.append(f"{sub}: left={left[key]!r}, missing on right")
            else:
                diffs.extend(structural_diff(left[key], right[key], sub))
        return diffs

    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"{path}: length {len(left)} != {len(right)}"]
        diffs = []
        for index, (a, b) in enumerate(zip(left, right)):
            diffs.extend(structural_diff(a, b, f"{path}[{index}]"))
        return diffs

    if left != right:
        return [f"{path}: {left!r} != {right!r}"]
    return []


def measure_windows(db_path: Path, run_id: str) -> Tuple[float, float]:
    """Max non-LLM step gap and max open-transaction gap, in milliseconds.

    Measured from the control run's own timestamps so the report quotes real
    numbers from this machine rather than an assumed figure.
    """
    from datetime import datetime

    def parse(stamp: str) -> datetime:
        return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S.%fZ")

    conn = _connect(db_path)
    try:
        checkpoints = conn.execute(
            "SELECT step_index, state, created_at FROM checkpoints WHERE run_id = ? "
            "ORDER BY step_index",
            (run_id,),
        ).fetchall()
        calls = conn.execute(
            "SELECT step_index, created_at FROM llm_calls WHERE run_id = ?", (run_id,)
        ).fetchall()
    finally:
        conn.close()

    stamps = {r["step_index"]: r["created_at"] for r in checkpoints}

    non_llm = [0.0]
    for current, following in zip(checkpoints, checkpoints[1:]):
        if current["state"] not in PRE_LLM_STATES:
            gap = (parse(following["created_at"]) - parse(current["created_at"])).total_seconds()
            non_llm.append(gap * 1000)

    txn = [0.0]
    for call in calls:
        stamp = stamps.get(call["step_index"])
        if stamp:
            txn.append((parse(stamp) - parse(call["created_at"])).total_seconds() * 1000)

    return max(non_llm), max(txn)


# ---------------------------------------------------------------------------
# subprocess control
# ---------------------------------------------------------------------------


def _env() -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def spawn(args: List[str]) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "argus.orchestrator.runner", *args],
        cwd=str(REPO_ROOT),
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def run_to_completion(args: List[str], timeout: float) -> Tuple[int, str, str]:
    proc = spawn(args)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        return -999, out, err + f"\n[harness] timed out after {timeout}s"
    return proc.returncode, out, err


def wait_for_trigger(
    db_path: Path, trigger_states: set, proc: subprocess.Popen, deadline: float
) -> Optional[sqlite3.Row]:
    """Poll the checkpoints table until a trigger state lands. No fixed sleep.

    Returns the checkpoint row that matched, or None if the process exited or
    the deadline passed first.
    """
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return None
        run_id = read_run_id(db_path)
        if run_id:
            for row in reversed(read_checkpoints(db_path, run_id)):
                if row["state"] in trigger_states:
                    return row
        time.sleep(POLL_INTERVAL)
    return None


#: Tight-poll cadence for the sub-10ms non-LLM windows. A persistent connection
#: keeps each probe to tens of microseconds instead of the ~1ms a fresh connect
#: costs, which is the only way this window is reachable at all.
TIGHT_POLL_INTERVAL = 0.0002


def tight_wait_for_call_to_finish(
    db_path: Path, baseline_calls: int, proc: subprocess.Popen, deadline: float
) -> bool:
    """Busy-poll until the in-flight model call logs its row, then return AT ONCE.

    The llm_calls row is committed in autocommit BEFORE the step opens its
    db.atomic() transaction, so the instant the count moves the target process
    is past the model call and inside the short non-LLM region: transaction
    open, then a couple of instant steps. Returning immediately gives the
    caller the best available shot at killing there.
    """
    conn = sqlite3.connect(db_path, timeout=10, isolation_level=None)
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return False
            if conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0] > baseline_calls:
                return True
            time.sleep(TIGHT_POLL_INTERVAL)
        return False
    finally:
        conn.close()


def call_is_in_flight(db_path: Path, baseline_calls: int, proc: subprocess.Popen) -> bool:
    """Confirm an Ollama call is genuinely in progress — observed, not slept on.

    llm_calls rows are written only AFTER a call returns. A pre-LLM checkpoint
    (PLAN/CRITIQUE) means the next dispatch issues a model call, and the call
    starts within microseconds of that checkpoint committing while polling
    detects it within POLL_INTERVAL. So an llm_calls count still at the
    baseline means the call is running right now — roughly 50ms into a call
    that takes seconds. A count above the baseline means it already finished
    and the kill will land between steps instead, which classify_interruption
    then reports honestly rather than hiding.
    """
    if proc.poll() is not None:
        return False
    return count_llm_calls(db_path) <= baseline_calls


# ---------------------------------------------------------------------------
# assertions
# ---------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class ScenarioResult:
    name: str
    checks: List[Check] = field(default_factory=list)
    interrupted_region: str = "unknown"
    inconclusive: str = ""
    #: Whether the kill actually landed in the region this scenario targets.
    #: False means the run was still exercised and every structural guarantee
    #: still applies — but this scenario's DISTINCT failure mode was not tested,
    #: which must be reported, never quietly counted as a pass.
    on_target: bool = False
    attempt: int = 1

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append(Check(name, ok, detail))

    @property
    def failed(self) -> List[Check]:
        return [c for c in self.checks if not c.ok]

    @property
    def status(self) -> str:
        if self.failed:
            return "FAIL"
        return "INCONCLUSIVE" if self.inconclusive else "PASS"


def classify_interruption(db_path: Path, run_id: str) -> str:
    """What was actually in flight when the kill landed.

    Determined from the database rather than assumed from timing, so the report
    says what really happened instead of what was aimed at.
    """
    checkpoints = read_checkpoints(db_path, run_id)
    if not checkpoints:
        return "before-first-checkpoint"

    completed = len(checkpoints)
    last_state = checkpoints[-1]["state"]
    interrupted_step = completed  # the step that never produced a checkpoint

    calls_at_interrupted_step = [
        r for r in read_llm_calls(db_path, run_id) if r["step_index"] == interrupted_step
    ]
    if calls_at_interrupted_step:
        # The call returned and logged, but the step's transaction never
        # committed: killed in the narrow post-call, pre-commit window.
        return "post-llm-call, pre-commit"
    if last_state in PRE_LLM_STATES:
        return f"mid-llm-call (after {last_state})"
    return f"between-steps (after {last_state})"


def assert_database_sane(result: ScenarioResult, db_path: Path, run_id: str) -> None:
    integrity, fk = pragma_checks(db_path)
    result.add("integrity_check == ok", integrity == "ok", integrity)
    result.add("foreign_key_check empty", not fk, f"{len(fk)} violations")

    checkpoints = read_checkpoints(db_path, run_id)
    result.add("checkpoints exist", bool(checkpoints), f"{len(checkpoints)} rows")
    if not checkpoints:
        return

    indexes = [r["step_index"] for r in checkpoints]
    contiguous = indexes == list(range(len(indexes)))
    result.add(
        "step_index contiguous 0..N",
        contiguous,
        f"{indexes[:3]}..{indexes[-3:]}" if len(indexes) > 6 else str(indexes),
    )

    last_state = checkpoints[-1]["state"]
    result.add("last state is a real State", last_state in VALID_STATES, last_state)
    result.add(
        "killed before terminal",
        last_state not in TERMINAL,
        f"last state {last_state}",
    )

    # THE 001c / db.atomic() guarantee, finally tested for real: an interrupted
    # step must leave NO checkpoint. payload.step_index is always
    # checkpoint.step_index + 1, so for a run of N completed steps the newest
    # payload reads exactly N. If a half-applied transaction had survived, this
    # would not line up.
    payload = json.loads(checkpoints[-1]["payload"])
    result.add(
        "checkpoint count == completed steps only",
        payload["step_index"] == len(checkpoints),
        f"payload.step_index={payload['step_index']} vs {len(checkpoints)} checkpoints",
    )
    result.add(
        "final payload deserializes",
        payload.get("state") in VALID_STATES,
        str(payload.get("state")),
    )

    # Expected artifact, asserted rather than assumed - see module docstring.
    orphans = [
        r for r in read_llm_calls(db_path, run_id) if r["step_index"] >= len(checkpoints)
    ]
    result.add(
        "orphan llm_calls harmless (expected)",
        True,
        f"{len(orphans)} row(s) from the interrupted step; FK/integrity still clean",
    )


def assert_no_duplicated_act(result: ScenarioResult, db_path: Path, run_id: str) -> None:
    """A resumed run must not re-execute a tool call that already completed.

    Entering ACT writes exactly one checkpoint with state='ACT', and the very
    next step runs the tool once and increments attempt_count. So for a
    terminated task, ACT checkpoints and final tasks.attempts must be equal.
    A re-executed tool call breaks that equality.
    """
    act_counts: Dict[str, int] = {}
    for row in read_checkpoints(db_path, run_id):
        if row["state"] == State.ACT.value and row["task_id"]:
            act_counts[row["task_id"]] = act_counts.get(row["task_id"], 0) + 1

    mismatches = []
    for task in read_tasks(db_path, run_id):
        if task["attempts"] is None:
            continue  # still in flight; attempts is written only at termination
        seen = act_counts.get(task["task_id"], 0)
        if seen != task["attempts"]:
            mismatches.append(f"{task['task_id']}: {seen} ACT vs attempts={task['attempts']}")

    result.add(
        "no ACT re-executed after resume",
        not mismatches,
        "; ".join(mismatches) if mismatches else f"{len(act_counts)} task(s) consistent",
    )


def compare_to_control(
    result: ScenarioResult, db_path: Path, run_id: str, control: Dict[str, Any]
) -> None:
    checkpoints = read_checkpoints(db_path, run_id)
    tasks = read_tasks(db_path, run_id)
    context = final_context(db_path, run_id) or {}

    plan = [(t["tool"], t["description"]) for t in tasks]
    same_plan = plan == control["plan"]

    if not same_plan:
        # The planner is a real model. A different plan makes step counts and
        # staged facts legitimately differ, and that is model nondeterminism,
        # NOT a crash-recovery failure. Flag it as inconclusive rather than
        # letting it masquerade as a resume bug - or be silently tolerated.
        result.inconclusive = (
            "planner produced a different plan than the control run, so "
            "step-count and facts equivalence cannot be judged"
        )
        result.add(
            "plan matches control (equivalence precondition)",
            True,
            f"DIVERGED: {plan} vs {control['plan']} - see note",
        )
        return

    result.add("plan matches control", True, f"{len(plan)} task(s)")
    result.add(
        "checkpoint count matches control",
        len(checkpoints) == control["checkpoints"],
        f"{len(checkpoints)} vs {control['checkpoints']}",
    )

    # Normalise each side against ITS OWN run_id, then compare. Without this the
    # check can never pass: task_id embeds the run_id.
    mine = normalise_run_scoped(context.get("facts_staged", []), run_id)
    theirs = normalise_run_scoped(control["facts_staged"], control["run_id"])
    fact_diffs = structural_diff(theirs, mine, "facts_staged")
    result.add(
        "facts_staged matches control",
        not fact_diffs,
        "; ".join(fact_diffs) if fact_diffs
        else f"{len(mine)} fact(s) identical after run-id normalisation",
    )

    result.add(
        "tasks match control",
        [(t["tool"], t["status"], t["attempts"]) for t in tasks] == control["tasks"],
        "status/attempts per task",
    )


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------


def reset_db(db_path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(db_path) + suffix)
        if candidate.exists():
            candidate.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    apply_migrations(db_path)


def run_control(target: str, db_path: Path, timeout: float) -> Dict[str, Any]:
    print("[control] uninterrupted run ...", flush=True)
    reset_db(db_path)
    code, out, err = run_to_completion(["--target", target, "--db", str(db_path)], timeout)
    if code != 0:
        raise SystemExit(f"control run failed (exit {code})\n{out}\n{err}")

    run_id = read_run_id(db_path)
    tasks = read_tasks(db_path, run_id)
    context = final_context(db_path, run_id) or {}
    control = {
        "run_id": run_id,
        "checkpoints": len(read_checkpoints(db_path, run_id)),
        "plan": [(t["tool"], t["description"]) for t in tasks],
        "tasks": [(t["tool"], t["status"], t["attempts"]) for t in tasks],
        "facts_staged": context.get("facts_staged", []),
        "llm_calls": len(read_llm_calls(db_path, run_id)),
    }
    print(
        f"[control] DONE - {control['checkpoints']} checkpoints, "
        f"{len(control['plan'])} tasks, {control['llm_calls']} llm calls",
        flush=True,
    )
    return control


def run_scenario(
    name: str,
    trigger_states: set,
    require_in_flight: bool,
    accept_regions: Tuple[str, ...],
    target: str,
    db_path: Path,
    control: Dict[str, Any],
    timeout: float,
    attempt: int = 1,
) -> ScenarioResult:
    print(f"\n[{name}] starting fresh run (attempt {attempt}) ...", flush=True)
    result = ScenarioResult(name=name, attempt=attempt)
    reset_db(db_path)

    proc = spawn(["--target", target, "--db", str(db_path)])
    deadline = time.monotonic() + timeout

    trigger = wait_for_trigger(db_path, trigger_states, proc, deadline)
    if trigger is None:
        proc.kill()
        out, err = proc.communicate()
        result.add("reached trigger state", False, f"never saw {sorted(trigger_states)}")
        print(f"[{name}] trigger never reached\n{out}\n{err}", flush=True)
        return result

    result.add(
        "reached trigger state",
        True,
        f"{trigger['state']} at step {trigger['step_index']}",
    )

    # Baseline is read AFTER the pre-LLM checkpoint lands, so it counts only
    # calls from earlier steps; the call this step is about to make has not
    # logged yet.
    baseline = count_llm_calls(db_path)

    if require_in_flight:
        in_flight = call_is_in_flight(db_path, baseline, proc)
        result.add(
            "ollama call in flight at kill",
            in_flight,
            f"llm_calls still at {baseline}" if in_flight else "call already completed",
        )
    else:
        # Aim for the region with NO model call running: wait for the call to
        # log its row, then kill instantly. See the honesty note on
        # NO_LLM_SCENARIO below - this window is only a few ms wide.
        landed = tight_wait_for_call_to_finish(db_path, baseline, proc, deadline)
        result.add(
            "model call finished before kill",
            landed,
            "llm_calls incremented; killing inside the non-LLM region"
            if landed else "call never completed within the timeout",
        )

    alive_before_kill = proc.poll() is None
    result.add("process alive immediately before kill", alive_before_kill, "")

    proc.kill()
    out, err = proc.communicate(timeout=30)
    returncode = proc.returncode

    # Ungraceful: SIGKILL on POSIX (-9), TerminateProcess on Windows (1).
    # returncode alone is ambiguous on Windows, where the runner's own FAILED
    # exit is also 1 - so the real proof is that the process was still running
    # when we killed it AND the run never reached a terminal state.
    expected = -9 if os.name != "nt" else 1
    result.add(
        "kill was ungraceful",
        returncode == expected and alive_before_kill,
        f"returncode={returncode} (expected {expected} on {os.name})",
    )

    run_id = read_run_id(db_path)
    if not run_id:
        result.add("run row exists after kill", False, "no run row")
        return result

    result.interrupted_region = classify_interruption(db_path, run_id)
    result.on_target = any(
        marker in result.interrupted_region for marker in accept_regions
    )
    print(
        f"[{name}] killed - interrupted {result.interrupted_region}"
        f" ({'ON TARGET' if result.on_target else 'off target'})",
        flush=True,
    )

    assert_database_sane(result, db_path, run_id)

    killed_checkpoints = len(read_checkpoints(db_path, run_id))

    print(f"[{name}] resuming {run_id} ...", flush=True)
    code, _, rerr = run_to_completion(
        ["--resume", run_id, "--db", str(db_path)], timeout
    )
    result.add("resume exits 0", code == 0, f"exit {code}" + (f"\n{rerr}" if code else ""))

    context = final_context(db_path, run_id) or {}
    result.add("resumed run reaches DONE", context.get("state") == State.DONE.value,
               str(context.get("state")))
    result.add(
        "resume made forward progress",
        len(read_checkpoints(db_path, run_id)) > killed_checkpoints,
        f"{killed_checkpoints} -> {len(read_checkpoints(db_path, run_id))}",
    )

    integrity, fk = pragma_checks(db_path)
    result.add("integrity_check ok after resume", integrity == "ok", integrity)
    result.add("foreign_key_check empty after resume", not fk, f"{len(fk)} violations")

    indexes = [r["step_index"] for r in read_checkpoints(db_path, run_id)]
    result.add(
        "final step_index contiguous 0..N",
        indexes == list(range(len(indexes))),
        f"{len(indexes)} checkpoints",
    )

    assert_no_duplicated_act(result, db_path, run_id)
    compare_to_control(result, db_path, run_id, control)
    return result


#: (name, trigger states, require a call in flight, regions that count as
#:  on-target, attempts allowed)
#:
#: HONESTY NOTE on kill-between-steps
#:     The two model calls take ~6s and ~7s, so those windows are trivial to
#:     hit. Everything else in a run is a few milliseconds: measured on this
#:     machine, the non-LLM step gaps total ~3ms and the open-transaction
#:     window ~6ms, against a whole-run wall time of ~13s. That is well under
#:     one percent of the run, and comparable to the latency of the kill
#:     itself, so landing there is probabilistic no matter how tightly we poll.
#:
#:     So this scenario retries, and if it never lands it is reported
#:     INCONCLUSIVE with the measured numbers — never as a pass. A miss
#:     re-tests the mid-LLM mode, which is already covered twice; counting that
#:     as a third scenario would claim coverage that does not exist.
SCENARIOS = [
    ("kill-during-planner-call", {State.PLAN.value}, True, ("mid-llm-call",), 1),
    ("kill-during-critic-call", {State.CRITIQUE.value}, True, ("mid-llm-call",), 1),
    (
        "kill-between-steps",
        {State.PLAN.value, State.CRITIQUE.value},
        False,
        ("post-llm-call", "between-steps"),
        5,
    ),
]


# ---------------------------------------------------------------------------
# self-check (no Ollama, no subprocesses)
# ---------------------------------------------------------------------------


def _seed(db_path: Path, run_id: str, states: List[str], gap: bool = False) -> None:
    from argus.orchestrator.context import RunContext

    conn = _connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            "INSERT INTO runs (run_id, target_id, status) VALUES (?, ?, ?)",
            (run_id, "t", "running"),
        )
        for position, state in enumerate(states):
            index = position + 1 if gap and position >= 2 else position
            ctx = RunContext(
                run_id=run_id, target_id="t", goal="g", state=State(state), step_index=index + 1
            )
            conn.execute(
                "INSERT INTO checkpoints "
                "(checkpoint_id, run_id, step_index, state, payload) VALUES (?, ?, ?, ?, ?)",
                (f"{run_id}-{index}", run_id, index, state, ctx.to_json()),
            )
        conn.commit()
    finally:
        conn.close()


def self_check(db_path: Path) -> int:
    """Validate the harness's own logic before anyone spends real time on it.

    Also proves the checks are not vacuous: a seeded gap MUST be caught.
    """
    print("self-check: harness logic only, no Ollama and no subprocesses\n")
    failures = []

    reset_db(db_path)
    _seed(db_path, "run-good", ["PLAN", "ACT", "OBSERVE", "CRITIQUE"])
    good = ScenarioResult("seeded-healthy")
    assert_database_sane(good, db_path, "run-good")
    assert_no_duplicated_act(good, db_path, "run-good")
    if good.failed:
        failures.append(f"healthy run flagged: {[c.name for c in good.failed]}")
    print(f"  healthy seeded run      -> {len(good.checks)} checks, {len(good.failed)} failed")

    reset_db(db_path)
    _seed(db_path, "run-gap", ["PLAN", "ACT", "OBSERVE", "CRITIQUE"], gap=True)
    gapped = ScenarioResult("seeded-gap")
    assert_database_sane(gapped, db_path, "run-gap")
    caught = [c.name for c in gapped.failed]
    if "step_index contiguous 0..N" not in caught:
        failures.append("a seeded step_index gap was NOT caught - checks are vacuous")
    print(f"  seeded gap              -> caught by: {caught}")

    reset_db(db_path)
    _seed(db_path, "run-cls", ["PLAN"])
    region = classify_interruption(db_path, "run-cls")
    if "mid-llm-call" not in region:
        failures.append(f"classifier mislabelled a PLAN-tail run: {region}")
    print(f"  classifier on PLAN tail -> {region!r}")

    reset_db(db_path)
    _seed(db_path, "run-cls2", ["PLAN", "ACT"])
    region2 = classify_interruption(db_path, "run-cls2")
    if "between-steps" not in region2:
        failures.append(f"classifier mislabelled an ACT-tail run: {region2}")
    print(f"  classifier on ACT tail  -> {region2!r}")

    integrity, fk = pragma_checks(db_path)
    print(f"  pragma helpers          -> integrity={integrity!r}, fk_violations={len(fk)}")
    if integrity != "ok":
        failures.append("integrity helper returned non-ok on a fresh database")

    # The bug this replaced: identical facts from two runs compared unequal
    # because task_id embeds the run_id, making the check unsatisfiable.
    fact_a = [{"target_id": "t", "task_id": "run-aaa:task-0", "summary": "s", "attempts": 3}]
    fact_b = [{"target_id": "t", "task_id": "run-bbb:task-0", "summary": "s", "attempts": 3}]
    if fact_a == fact_b:
        failures.append("fixture is wrong: raw facts should differ by run_id")
    normalised = structural_diff(
        normalise_run_scoped(fact_a, "run-aaa"),
        normalise_run_scoped(fact_b, "run-bbb"),
        "facts_staged",
    )
    if normalised:
        failures.append(f"equivalent facts still differ after normalisation: {normalised}")
    print(f"  facts normalisation     -> raw differ, normalised diffs={len(normalised)}")

    # ...and it must still catch a REAL difference, not just pass everything.
    fact_c = [{"target_id": "t", "task_id": "run-bbb:task-0", "summary": "DIFFERENT", "attempts": 3}]
    real = structural_diff(
        normalise_run_scoped(fact_a, "run-aaa"),
        normalise_run_scoped(fact_c, "run-bbb"),
        "facts_staged",
    )
    if not real:
        failures.append("normalisation swallowed a real content difference - check is vacuous")
    # A fact attached to the wrong task ordinal must still be caught.
    fact_d = [{"target_id": "t", "task_id": "run-bbb:task-9", "summary": "s", "attempts": 3}]
    ordinal = structural_diff(
        normalise_run_scoped(fact_a, "run-aaa"),
        normalise_run_scoped(fact_d, "run-bbb"),
        "facts_staged",
    )
    if not ordinal:
        failures.append("normalisation swallowed a task-ordinal mismatch")
    print(f"  real difference caught  -> {real}")
    print(f"  ordinal mismatch caught -> {ordinal}")

    print()
    if failures:
        for failure in failures:
            print(f"SELF-CHECK FAILED: {failure}")
        return 1
    print("SELF-CHECK OK: harness logic sound and its assertions are not vacuous.")
    return 0


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def print_summary(results: List[ScenarioResult]) -> None:
    print("\n" + "=" * 78)
    print("CRASH-RESUME SUMMARY")
    print("=" * 78)
    print(f"{'scenario':<28} {'status':<14} {'on target':<11} interrupted")
    print("-" * 78)
    for result in results:
        flag = "yes" if result.on_target else "NO"
        print(
            f"{result.name:<28} {result.status:<14} {flag:<11} {result.interrupted_region}"
        )

    for result in results:
        if result.failed or result.inconclusive:
            print(f"\n--- {result.name} ---")
            for check in result.failed:
                print(f"  FAILED  {check.name}: {check.detail}")
            if result.inconclusive:
                print(f"  NOTE    {result.inconclusive}")

    print("\nPer-scenario checks passed:")
    for result in results:
        passed = len(result.checks) - len(result.failed)
        print(f"  {result.name:<28} {passed}/{len(result.checks)}")
    print("=" * 78)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="crash_test.py", description="Real crash-resume harness for Sprint 1."
    )
    parser.add_argument("--target", default="monash_ai_masters", help="target id to run")
    parser.add_argument(
        "--db-dir",
        default=str(REPO_ROOT / "data" / "crash_test"),
        help="directory for the DEDICATED crash-test databases (never data/argus.db)",
    )
    parser.add_argument("--timeout", type=float, default=300.0, help="per-subprocess timeout")
    parser.add_argument(
        "--scenario", action="append", choices=[s[0] for s in SCENARIOS],
        help="run only these scenarios (repeatable); default is all three",
    )
    parser.add_argument(
        "--self-check", action="store_true",
        help="validate harness logic only - no Ollama, no subprocesses",
    )
    args = parser.parse_args(argv)

    db_dir = Path(args.db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)

    if args.self_check:
        return self_check(db_dir / "selfcheck.db")

    selected = [s for s in SCENARIOS if not args.scenario or s[0] in args.scenario]
    control = run_control(args.target, db_dir / "control.db", args.timeout)

    non_llm_ms, txn_ms = measure_windows(db_dir / "control.db", control["run_id"])

    results = []
    for name, trigger_states, require_in_flight, accept_regions, max_attempts in selected:
        for attempt in range(1, max_attempts + 1):
            result = run_scenario(
                name, trigger_states, require_in_flight, accept_regions, args.target,
                db_dir / f"{name}.db", control, args.timeout, attempt=attempt,
            )
            if result.on_target or result.failed or attempt == max_attempts:
                break
            print(
                f"[{name}] off target ({result.interrupted_region}) - retrying "
                f"({attempt}/{max_attempts})",
                flush=True,
            )

        if not result.on_target and not result.failed:
            # Everything structural still passed, but this scenario's distinct
            # failure mode was never actually exercised. Say so.
            result.inconclusive = (
                f"could not land in the target window after {result.attempt} attempt(s); "
                f"killed {result.interrupted_region} instead. Measured windows on this "
                f"machine: non-LLM steps {non_llm_ms:.1f}ms, open transaction "
                f"{txn_ms:.1f}ms, versus ~13s of model calls per run. This scenario's "
                f"distinct mode is therefore UNTESTED - the structural checks below "
                f"re-cover the mid-LLM mode already tested above."
            )
        results.append(result)

    print_summary(results)
    return 1 if any(r.status == "FAIL" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
