"""Orchestrator state machine tests.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from argus import config  # noqa: E402
from argus.orchestrator import fake_llm, llm, loop  # noqa: E402
from argus.orchestrator.context import RunContext  # noqa: E402
from argus.orchestrator.state import State  # noqa: E402
from argus.storage import db  # noqa: E402
from argus.storage.migrate import apply_migrations  # noqa: E402


class FakeChatResponse:
    """Shaped like ollama's ChatResponse: message.content plus timing fields."""

    class _Message:
        def __init__(self, content: str) -> None:
            self.content = content
            self.thinking = None

    def __init__(self, content: str, eval_count: int = 120) -> None:
        self.message = self._Message(content)
        self.prompt_eval_count = 350
        self.eval_count = eval_count
        self.eval_duration = 5_000_000_000  # 5s -> 24 tok/s at eval_count=120
        self.total_duration = 5_200_000_000
        self.load_duration = 120_000_000
        self.done = True
        self.done_reason = "stop"


class MockOllama:
    """Stands in for llm._invoke, the single seam over the network.

    Reproduces what the old fakes produced so the existing assertions still
    mean the same thing: a 2-task plan, and a critic that fails twice per task
    then passes. The critic counter is per-call rather than derived from the
    prompt, because critic.md deliberately withholds run history — the prompt
    is byte-identical across retries of the same action, so there is nothing in
    it to count.
    """

    def __init__(self) -> None:
        self.calls: list = []
        self._critic_calls = 0

    def __call__(self, model_tag, messages, schema):
        prompt = messages[0]["content"]
        self.calls.append({"model_tag": model_tag, "messages": messages, "schema": schema})

        if "You are the planner" in prompt:
            return FakeChatResponse(
                json.dumps(
                    {
                        "thought": "fetch the page, then confirm it parsed",
                        "done": False,
                        "tasks": [
                            {
                                "tool_name": "stub_fetch_ok",
                                "args": {"url": "https://example.invalid/a"},
                                "externally_visible": False,
                            },
                            {
                                "tool_name": "stub_fetch_ok",
                                "args": {"url": "https://example.invalid/b"},
                                "externally_visible": False,
                            },
                        ],
                    }
                )
            )

        if "You are the critic" in prompt:
            self._critic_calls += 1
            passes = self._critic_calls % (fake_llm.CRITIQUE_FAILURES_PER_TASK + 1) == 0
            return FakeChatResponse(
                json.dumps(
                    {
                        "verdict": "pass" if passes else "fail",
                        "confidence": 0.9 if passes else 0.4,
                        "reflection": "" if passes else "narrow the extraction and retry",
                    }
                )
            )

        raise AssertionError(f"unexpected prompt sent to Ollama:\n{prompt[:200]}")


def checkpoint_rows(db_path: Path, run_id: str) -> List[Any]:
    with db.connect(db_path) as conn:
        return conn.execute(
            "SELECT step_index, state FROM checkpoints WHERE run_id = ? "
            "ORDER BY step_index",
            (run_id,),
        ).fetchall()


def step_indexes(db_path: Path, run_id: str) -> List[int]:
    return [row["step_index"] for row in checkpoint_rows(db_path, run_id)]


def task_rows(db_path: Path, run_id: str) -> List[Any]:
    with db.connect(db_path) as conn:
        return conn.execute(
            "SELECT task_id, status, attempts FROM tasks WHERE run_id = ? "
            "ORDER BY task_index",
            (run_id,),
        ).fetchall()


def oversized_batch_planner(ctx: RunContext, conn: Any = None) -> Dict[str, Any]:
    """A planner that asks for more tasks than MAX_TASKS_PER_RUN in ONE call.

    Under the literal transition table, the planner is SINGLE-SHOT per run:
    PERSIST -> PLAN only fires when the local queue is non-empty, and PLAN's
    "queue is empty, ask the planner" branch only fires when it's empty. Those
    conditions are mutually exclusive, so once a batch is handed back it is the
    only batch — confirmed empirically (a stub planner returning done=True on a
    hypothetical second call was never called a second time). A "runaway
    planner" in this design therefore means an oversized single batch, not
    repeated re-invocation.

    Batch size is a FIXED literal (25), deliberately not derived from
    loop.MAX_TASKS_PER_RUN: the step-backstop test patches that constant to a
    huge value so the guard won't trip, and a batch size tracking it live would
    grow to match, tripping the (patched) guard anyway and defeating the point
    of that test. 25 exceeds the real default (20) for the guard test, and is
    trivially under any patched value for the backstop test.
    """
    return {
        "done": False,
        "parse_ok": True,
        "tasks": [
            {
                "description": f"task {index}",
                "tool": "open_page",
                "action": {
                    "tool": "open_page",
                    "args": {},
                    "externally_visible": False,
                },
            }
            for index in range(25)
        ],
    }


def externally_visible_planner(ctx: RunContext, conn: Any = None) -> Dict[str, Any]:
    """One task whose action leaves the machine — the Sprint 5 gate branch."""
    if ctx.facts_staged or ctx.current_task is not None:
        return {"done": True, "parse_ok": True, "tasks": []}
    return {
        "done": False,
        "parse_ok": True,
        "tasks": [
            {
                "description": "send the digest email",
                "tool": "send_email",
                "action": {
                    "tool": "send_email",
                    "args": {"to": "someone@example.invalid"},
                    "externally_visible": True,
                },
            }
        ],
    }


class LoopTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "test_argus.db"
        self.addCleanup(self._tmp.cleanup)
        apply_migrations(self.db_path)

        # Patch the ONE network seam. Everything above it — prompt rendering,
        # schema validation, llm_calls logging, the adapters — runs for real.
        self.ollama = MockOllama()
        patcher = mock.patch.object(llm, "_invoke", self.ollama)
        patcher.start()
        self.addCleanup(patcher.stop)


class TestContextRoundTrip(unittest.TestCase):
    def test_json_round_trip_is_lossless(self) -> None:
        ctx = RunContext(
            run_id="run-1",
            target_id="monash_ai_masters",
            goal="check for changes",
            state=State.CRITIQUE,
            step_index=7,
            task_queue=[{"task_id": "run-1:task-1", "task_index": 1}],
            current_task={"task_id": "run-1:task-0", "task_index": 0},
            attempt_count=2,
            parse_retries=1,
            last_action={"tool": "open_page", "externally_visible": False},
            last_observation={"ok": True, "content": "stub"},
            reflections=["first attempt was too broad"],
            facts_staged=[{"summary": "intake dates unchanged"}],
        )

        restored = RunContext.from_json(ctx.to_json())

        self.assertEqual(restored, ctx)
        self.assertIs(restored.state, State.CRITIQUE)

    def test_state_serializes_as_its_value(self) -> None:
        """str(State.PLAN) is 'State.PLAN', not 'PLAN' — the payload must carry
        the value, or resume rebuilds a different enum member.
        """
        ctx = RunContext(run_id="r", target_id="t", goal="g", state=State.PLAN)
        self.assertEqual(ctx.to_dict()["state"], "PLAN")


class TestPureDataModules(unittest.TestCase):
    """state.py and context.py are pure data: enum and dataclass, no I/O.

    Only loop.py touches the database. This keeps the transition table testable
    without a connection and lets Sprint 6's dashboard reuse the types.
    Checks the import statements via AST rather than sys.modules, because other
    tests import loop.py (which legitimately imports storage) and would pollute
    a sys.modules-based check.
    """

    def _imported_modules(self, path: Path) -> List[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
        return names

    def test_state_does_not_import_storage(self) -> None:
        imported = self._imported_modules(SRC / "argus" / "orchestrator" / "state.py")
        self.assertFalse([n for n in imported if "storage" in n], imported)

    def test_context_does_not_import_storage(self) -> None:
        imported = self._imported_modules(SRC / "argus" / "orchestrator" / "context.py")
        self.assertFalse([n for n in imported if "storage" in n], imported)

    def test_loop_does_import_storage(self) -> None:
        """Control: proves the check above can actually detect an import."""
        imported = self._imported_modules(SRC / "argus" / "orchestrator" / "loop.py")
        self.assertTrue([n for n in imported if "storage" in n], imported)


class TestFullRun(LoopTestCase):
    def test_run_reaches_done_with_contiguous_step_indexes(self) -> None:
        ctx = loop.new_run("test_target")
        loop.run_to_completion(ctx, db_path=self.db_path)

        self.assertIs(ctx.state, State.DONE)

        indexes = step_indexes(self.db_path, ctx.run_id)
        # sorted(...) == range(n) catches gaps AND duplicates in one assertion.
        self.assertEqual(sorted(indexes), list(range(len(indexes))))
        self.assertEqual(ctx.step_index, len(indexes))
        self.assertEqual(checkpoint_rows(self.db_path, ctx.run_id)[-1]["state"], "DONE")

    def test_both_tasks_terminate_done_with_attempts_recorded(self) -> None:
        ctx = loop.new_run("test_target")
        loop.run_to_completion(ctx, db_path=self.db_path)

        rows = task_rows(self.db_path, ctx.run_id)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["status"], "done")
            self.assertEqual(row["attempts"], fake_llm.CRITIQUE_FAILURES_PER_TASK + 1)


class TestAttemptsLifecycle(LoopTestCase):
    def test_attempts_is_null_in_flight_and_set_once_terminal(self) -> None:
        """001c's CHECK ties terminal status to a non-null attempts. This walks
        a task from pending through running to done and asserts the pairing at
        each stage rather than only at the end.
        """
        ctx = loop.new_run("test_target")

        with db.connect(self.db_path) as conn:
            # INIT -> PLAN -> ACT: the task now exists and is in flight.
            loop.step(ctx, conn=conn)
            loop.step(ctx, conn=conn)
            self.assertIs(ctx.state, State.ACT)

            rows = task_rows(self.db_path, ctx.run_id)
            self.assertEqual(rows[0]["status"], "pending")
            self.assertIsNone(rows[0]["attempts"])

            loop.step(ctx, conn=conn)  # ACT -> OBSERVE
            rows = task_rows(self.db_path, ctx.run_id)
            self.assertEqual(rows[0]["status"], "running")
            self.assertIsNone(rows[0]["attempts"])

            # Drive the first task to termination.
            first_task_id = rows[0]["task_id"]
            while True:
                terminal = [
                    r for r in task_rows(self.db_path, ctx.run_id)
                    if r["task_id"] == first_task_id and r["status"] == "done"
                ]
                if terminal or ctx.state in loop.TERMINAL_STATES:
                    break
                loop.step(ctx, conn=conn)

        done_row = [
            r for r in task_rows(self.db_path, ctx.run_id) if r["task_id"] == first_task_id
        ][0]
        self.assertEqual(done_row["status"], "done")
        self.assertIsNotNone(done_row["attempts"])


class TestCritiqueRetry(LoopTestCase):
    def test_exactly_two_retries_then_pass(self) -> None:
        ctx = loop.new_run("test_target")

        act_count = 0
        with db.connect(self.db_path) as conn:
            while ctx.state is not State.GATE:
                loop.step(ctx, conn=conn)
                if ctx.state is State.ACT:
                    act_count += 1

        # 3 ACTs for the first task: initial attempt + 2 retries.
        self.assertEqual(act_count, fake_llm.CRITIQUE_FAILURES_PER_TASK + 1)
        self.assertEqual(ctx.attempt_count, fake_llm.CRITIQUE_FAILURES_PER_TASK + 1)

    def test_attempt_count_resets_for_the_second_task(self) -> None:
        """Reset per task is what makes fake_critique's attempt_count-derived
        verdict behave per task_id — and what keeps it resume-safe.
        """
        ctx = loop.new_run("test_target")
        loop.run_to_completion(ctx, db_path=self.db_path)

        rows = task_rows(self.db_path, ctx.run_id)
        self.assertEqual(
            [r["attempts"] for r in rows],
            [fake_llm.CRITIQUE_FAILURES_PER_TASK + 1] * 2,
        )


class TestResume(LoopTestCase):
    def _uninterrupted_total(self) -> int:
        with tempfile.TemporaryDirectory() as other:
            other_db = Path(other) / "baseline.db"
            apply_migrations(other_db)
            ctx = loop.new_run("test_target")
            loop.run_to_completion(ctx, db_path=other_db)
            self.assertIs(ctx.state, State.DONE)
            return len(step_indexes(other_db, ctx.run_id))

    def test_resume_once_matches_uninterrupted(self) -> None:
        baseline = self._uninterrupted_total()

        ctx = loop.new_run("test_target")
        run_id = ctx.run_id
        loop.run_to_completion(ctx, db_path=self.db_path, max_steps=5)
        self.assertNotIn(ctx.state, loop.TERMINAL_STATES)
        del ctx  # the in-memory context is gone, as after a crash

        resumed = loop.load_latest_checkpoint(run_id, db_path=self.db_path)
        self.assertIsNotNone(resumed)
        loop.run_to_completion(resumed, db_path=self.db_path)

        self.assertIs(resumed.state, State.DONE)
        indexes = step_indexes(self.db_path, run_id)
        self.assertEqual(sorted(indexes), list(range(len(indexes))))
        self.assertEqual(len(indexes), baseline)

    def test_resume_twice_matches_uninterrupted(self) -> None:
        """The resume-of-a-resume path, where an off-by-one in the step index
        invariant shows up as a duplicate or skipped index.
        """
        baseline = self._uninterrupted_total()

        ctx = loop.new_run("test_target")
        run_id = ctx.run_id
        loop.run_to_completion(ctx, db_path=self.db_path, max_steps=4)
        del ctx

        first = loop.load_latest_checkpoint(run_id, db_path=self.db_path)
        loop.run_to_completion(first, db_path=self.db_path, max_steps=6)
        self.assertNotIn(first.state, loop.TERMINAL_STATES)
        del first

        second = loop.load_latest_checkpoint(run_id, db_path=self.db_path)
        loop.run_to_completion(second, db_path=self.db_path)

        self.assertIs(second.state, State.DONE)
        indexes = step_indexes(self.db_path, run_id)
        self.assertEqual(sorted(indexes), list(range(len(indexes))))
        self.assertEqual(len(indexes), baseline)

    def test_resume_restores_full_context(self) -> None:
        ctx = loop.new_run("test_target")
        run_id = ctx.run_id
        loop.run_to_completion(ctx, db_path=self.db_path, max_steps=6)

        resumed = loop.load_latest_checkpoint(run_id, db_path=self.db_path)

        self.assertEqual(resumed, ctx)


class TestLLMWiring(LoopTestCase):
    """The loop really goes through llm.py, not the old fakes."""

    def test_full_run_logs_planner_and_critic_calls(self) -> None:
        ctx = loop.new_run("monash_ai_masters")
        loop.run_to_completion(ctx, db_path=self.db_path)
        self.assertIs(ctx.state, State.DONE)

        with db.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT role, model_tag, think_mode, prompt_file, ok, timed_out, "
                "tok_s, task_id, step_index FROM llm_calls WHERE run_id = ? "
                "ORDER BY rowid",
                (ctx.run_id,),
            ).fetchall()

        roles = [r["role"] for r in rows]
        # Planner is single-shot; critic runs once per attempt, 3 per task.
        self.assertEqual(roles.count("planner"), 1)
        self.assertEqual(roles.count("critic"), 6)

        for row in rows:
            self.assertEqual(row["think_mode"], "False")
            self.assertEqual(row["ok"], 1)
            self.assertEqual(row["timed_out"], 0)
            self.assertEqual(row["model_tag"], config.model_for_role(row["role"]))
            self.assertAlmostEqual(row["tok_s"], 24.0)
            self.assertIsNotNone(row["step_index"])

        # The planner runs before any task exists, so its row has no task_id.
        self.assertIsNone(rows[0]["task_id"])
        self.assertEqual(rows[0]["role"], "planner")

    def test_act_runs_the_real_stub_tools(self) -> None:
        ctx = loop.new_run("monash_ai_masters")
        loop.run_to_completion(ctx, db_path=self.db_path)

        # The planner asked for stub_fetch_ok, so the staged fact carries that
        # tool's real content rather than a canned observation string.
        self.assertIn("intake dates", ctx.facts_staged[0]["summary"])

    def test_failing_and_empty_tools_come_back_as_observations(self) -> None:
        """ACT -> OBSERVE is unconditional: neither a raising tool nor an empty
        result may escape as an exception.
        """
        from argus.orchestrator import tools

        failed = tools.run_tool("stub_fetch_fails", {})
        self.assertFalse(failed["ok"])
        self.assertIn("503", failed["error"])

        empty = tools.run_tool("stub_fetch_empty", {})
        self.assertTrue(empty["ok"])
        self.assertEqual(empty["content"], "")

        unknown = tools.run_tool("no_such_tool", {})
        self.assertFalse(unknown["ok"])
        self.assertIn("unknown tool", unknown["error"])


class TestCircuitBreaker(LoopTestCase):
    def test_task_guard_fails_the_run_on_an_oversized_batch(self) -> None:
        """The PRIMARY guard: caught at PLAN when the planner's (single-shot)
        batch alone exceeds MAX_TASKS_PER_RUN. Must trip well inside the step
        backstop's slack — that is what makes it the primary guard and not
        just an earlier copy of the same check.
        """
        stubs = loop.Stubs(plan=oversized_batch_planner)
        ctx = loop.new_run("test_target")
        loop.run_to_completion(ctx, db_path=self.db_path, stubs=stubs)

        self.assertIs(ctx.state, State.FAILED)
        self.assertTrue(
            any("MAX_TASKS_PER_RUN" in r for r in ctx.reflections), ctx.reflections
        )

        # The batch was rejected before any of its tasks were inserted.
        self.assertEqual(len(task_rows(self.db_path, ctx.run_id)), 0)
        self.assertLess(ctx.step_index, loop.MAX_STEPS)

    def test_stepping_a_terminated_run_raises(self) -> None:
        stubs = loop.Stubs(plan=oversized_batch_planner)
        ctx = loop.new_run("test_target")
        loop.run_to_completion(ctx, db_path=self.db_path, stubs=stubs)

        with self.assertRaises(ValueError):
            loop.step(ctx, db_path=self.db_path, stubs=stubs)


class TestStepBackstop(LoopTestCase):
    def test_step_backstop_fires_when_task_guard_is_bypassed(self) -> None:
        """Proves the step cap is a real, independent backstop and not dead
        code shadowed by the task guard: with MAX_TASKS_PER_RUN patched high
        enough to never trip, a large-but-now-allowed batch must still be
        stopped by MAX_STEPS alone partway through processing it. MAX_STEPS is
        also patched down so the test runs fast rather than taking hundreds of
        steps to process the full batch.
        """
        stubs = loop.Stubs(plan=oversized_batch_planner)
        ctx = loop.new_run("test_target")

        with mock.patch.object(loop, "MAX_TASKS_PER_RUN", 10**6), mock.patch.object(
            loop, "MAX_STEPS", 30
        ):
            loop.run_to_completion(ctx, db_path=self.db_path, stubs=stubs)

        self.assertIs(ctx.state, State.FAILED)
        self.assertTrue(
            any("step backstop" in r for r in ctx.reflections), ctx.reflections
        )

        rows = checkpoint_rows(self.db_path, ctx.run_id)
        # 30 work steps (indices 0..29) + 1 terminal row at index 30.
        self.assertEqual(len(rows), 31)
        self.assertEqual([r["step_index"] for r in rows], list(range(31)))
        self.assertEqual(rows[-1]["step_index"], 30)
        self.assertEqual(rows[-1]["state"], "FAILED")


class TestGateExternallyVisible(LoopTestCase):
    def test_externally_visible_action_fails_the_run(self) -> None:
        """Sprint 5 replaces this branch with AWAIT_APPROVAL. Until then it must
        be reachable and loggable rather than silently absent.
        """
        stubs = loop.Stubs(plan=externally_visible_planner)
        ctx = loop.new_run("test_target")
        loop.run_to_completion(ctx, db_path=self.db_path, stubs=stubs)

        self.assertIs(ctx.state, State.FAILED)

        rows = task_rows(self.db_path, ctx.run_id)
        self.assertEqual(rows[0]["status"], "failed")
        self.assertIsNotNone(rows[0]["attempts"])
        self.assertTrue(
            any("externally visible" in r for r in ctx.reflections), ctx.reflections
        )

        with db.connect(self.db_path) as conn:
            error = conn.execute(
                "SELECT error FROM tasks WHERE task_id = ?", (rows[0]["task_id"],)
            ).fetchone()["error"]
        self.assertIn("approval", error)


if __name__ == "__main__":
    unittest.main(verbosity=2)
