"""Wiring tests for the real Ollama calls.

These patch the ollama Client itself, one level BELOW llm._invoke, so the
parameters actually handed to client.chat are inspected — think, keep_alive,
format, options and the model tag. Mocking _invoke would skip exactly the
part these tests exist to check.

No Ollama process is contacted.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import httpx

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from argus import config  # noqa: E402
from argus.orchestrator import llm  # noqa: E402
from argus.orchestrator.context import RunContext  # noqa: E402
from argus.storage import db  # noqa: E402
from argus.storage.migrate import apply_migrations  # noqa: E402

VALID_PLAN = json.dumps(
    {
        "thought": "one fetch is enough",
        "done": False,
        "tasks": [
            {
                "tool_name": "stub_fetch_ok",
                "args": {"url": "https://example.invalid"},
                "externally_visible": False,
            }
        ],
    }
)


class FakeMessage:
    def __init__(self, content):
        self.content = content
        self.thinking = None


class FakeResponse:
    def __init__(self, content, eval_count=240, eval_duration=10_000_000_000):
        self.message = FakeMessage(content)
        self.prompt_eval_count = 512
        self.eval_count = eval_count
        self.eval_duration = eval_duration
        self.total_duration = eval_duration + 200_000_000
        self.load_duration = 120_000_000


class RecordingClient:
    """Captures constructor and chat kwargs; returns queued responses."""

    instances: list = []

    def __init__(self, host=None, timeout=None, **kwargs):
        self.host = host
        self.timeout = timeout
        self.chat_calls: list = []
        RecordingClient.instances.append(self)

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        response = RecordingClient.queue.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class LLMTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "llm.db"
        self.addCleanup(self._tmp.cleanup)
        apply_migrations(self.db_path)

        self.ctx = RunContext(run_id="run-1", target_id="t", goal="check t for changes")
        with db.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO runs (run_id, target_id, status) VALUES (?, ?, ?)",
                ("run-1", "t", "running"),
            )

        RecordingClient.instances = []
        RecordingClient.queue = []
        patcher = mock.patch.object(llm, "Client", RecordingClient)
        patcher.start()
        self.addCleanup(patcher.stop)

        # _get_client is lru_cached, so a client built under a previous patch
        # would survive into this test. Clear on the way in AND out.
        llm._get_client.cache_clear()
        self.addCleanup(llm._get_client.cache_clear)

    def plan(self, conn):
        return llm.call_ollama(
            "planner",
            "planner.md",
            {"goal": self.ctx.goal, "target_id": "t", "tool_catalog": "-", "max_tasks": 5},
            ctx=self.ctx,
            conn=conn,
        )

    def llm_rows(self):
        with db.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT * FROM llm_calls WHERE run_id = ? ORDER BY created_at, rowid",
                ("run-1",),
            ).fetchall()


class TestCallParameters(LLMTestCase):
    def test_think_is_false_and_top_level_not_in_options(self) -> None:
        """CLAUDE.md's sharpest convention: `think` is a TOP-LEVEL field. Inside
        options it is silently ignored and thinking stays on, which has been
        measured at 1583s returning empty content.
        """
        RecordingClient.queue = [FakeResponse(VALID_PLAN)]
        with db.connect(self.db_path) as conn:
            self.plan(conn)

        kwargs = RecordingClient.instances[0].chat_calls[0]
        self.assertIn("think", kwargs)
        self.assertIs(kwargs["think"], False)
        self.assertNotIn("think", kwargs["options"])

    def test_keep_alive_timeout_format_and_temperature(self) -> None:
        RecordingClient.queue = [FakeResponse(VALID_PLAN)]
        with db.connect(self.db_path) as conn:
            self.plan(conn)

        client = RecordingClient.instances[0]
        kwargs = client.chat_calls[0]

        self.assertEqual(client.timeout, 120)
        self.assertEqual(kwargs["keep_alive"], "30m")
        self.assertEqual(kwargs["options"]["temperature"], 0)
        self.assertEqual(kwargs["format"], llm.PlannerOutput.model_json_schema())
        self.assertNotIn("num_predict", kwargs["options"])

    def test_model_tag_comes_from_config(self) -> None:
        RecordingClient.queue = [FakeResponse(VALID_PLAN)]
        with db.connect(self.db_path) as conn:
            self.plan(conn)

        kwargs = RecordingClient.instances[0].chat_calls[0]
        self.assertEqual(kwargs["model"], config.model_for_role("planner"))

    def test_critic_prompt_excludes_goal_and_history(self) -> None:
        """A critic that can see the plan rubber-stamps it."""
        self.ctx.last_action = {"tool": "stub_fetch_ok", "args": {"url": "u"}}
        self.ctx.last_observation = {"ok": True, "content": "something"}
        self.ctx.reflections = ["an earlier reflection that must not leak"]
        RecordingClient.queue = [
            FakeResponse(json.dumps({"verdict": "pass", "confidence": 0.9, "reflection": ""}))
        ]

        with db.connect(self.db_path) as conn:
            llm.real_critique(self.ctx, conn)

        prompt = RecordingClient.instances[0].chat_calls[0]["messages"][0]["content"]
        self.assertNotIn(self.ctx.goal, prompt)
        self.assertNotIn("an earlier reflection that must not leak", prompt)
        self.assertIn("stub_fetch_ok", prompt)


class TestLogging(LLMTestCase):
    def test_successful_call_logs_one_row_with_tok_s(self) -> None:
        RecordingClient.queue = [FakeResponse(VALID_PLAN, eval_count=240, eval_duration=10_000_000_000)]
        with db.connect(self.db_path) as conn:
            self.plan(conn)

        rows = self.llm_rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["role"], "planner")
        self.assertEqual(row["prompt_file"], "planner.md")
        self.assertEqual(row["think_mode"], "False")
        self.assertEqual(row["ok"], 1)
        self.assertEqual(row["timed_out"], 0)
        self.assertIsNone(row["error"])
        self.assertEqual(row["eval_count"], 240)
        # Same formula as scripts/probe_thinking.py: 240 / (10e9 / 1e9) = 24.0
        self.assertAlmostEqual(row["tok_s"], 24.0)
        self.assertIn("stub_fetch_ok", row["raw_response"])

    def test_timeout_still_writes_a_row(self) -> None:
        """A timed-out call must never skip the insert — the 120s it cost is
        exactly what needs to be visible afterwards.
        """
        RecordingClient.queue = [httpx.ReadTimeout("timed out")]
        with db.connect(self.db_path) as conn:
            result = self.plan(conn)

        self.assertIsNone(result)
        rows = self.llm_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["timed_out"], 1)
        self.assertEqual(rows[0]["ok"], 0)
        self.assertIn("timeout", rows[0]["error"])
        self.assertEqual(self.ctx.parse_retries, 1)

    def test_timeout_is_not_retried(self) -> None:
        """Restating the schema cannot fix an unreachable server, so a timeout
        gets one attempt, not two.
        """
        RecordingClient.queue = [httpx.ReadTimeout("timed out")]
        with db.connect(self.db_path) as conn:
            self.plan(conn)

        self.assertEqual(len(RecordingClient.instances[0].chat_calls), 1)


class TestSchemaRetry(LLMTestCase):
    def test_invalid_then_valid_retries_once_and_succeeds(self) -> None:
        # `done` is required and absent -> a genuine ValidationError, not a
        # field that quietly defaults.
        RecordingClient.queue = [
            FakeResponse('{"thought": "no done field"}'),
            FakeResponse(VALID_PLAN),
        ]

        with db.connect(self.db_path) as conn:
            result = self.plan(conn)

        self.assertIsNotNone(result)
        self.assertEqual(len(RecordingClient.instances[0].chat_calls), 2)

        rows = self.llm_rows()
        self.assertEqual(len(rows), 2, "every attempt gets its own row")
        self.assertEqual(rows[0]["ok"], 0)
        self.assertIn("schema validation failed", rows[0]["error"])
        self.assertEqual(rows[1]["ok"], 1)
        self.assertEqual(self.ctx.parse_retries, 0)

    def test_retry_feeds_the_validation_error_back(self) -> None:
        RecordingClient.queue = [
            FakeResponse('{"thought": "no done field"}'),
            FakeResponse(VALID_PLAN),
        ]
        with db.connect(self.db_path) as conn:
            self.plan(conn)

        second_messages = RecordingClient.instances[0].chat_calls[1]["messages"]
        self.assertEqual(len(second_messages), 3)
        self.assertEqual(second_messages[1]["role"], "assistant")
        self.assertIn("failed schema validation", second_messages[2]["content"])

    def test_two_failures_return_none_and_increment_parse_retries(self) -> None:
        RecordingClient.queue = [
            FakeResponse('{"thought": "bad"}'),
            FakeResponse("not json at all"),
        ]
        with db.connect(self.db_path) as conn:
            result = self.plan(conn)

        self.assertIsNone(result)
        self.assertEqual(self.ctx.parse_retries, 1)
        rows = self.llm_rows()
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["ok"] == 0 for r in rows))

    def test_planner_failure_surfaces_as_parse_not_ok(self) -> None:
        RecordingClient.queue = [
            FakeResponse('{"thought": "bad"}'),
            FakeResponse("still bad"),
        ]
        with db.connect(self.db_path) as conn:
            result = llm.real_plan(self.ctx, conn)

        self.assertFalse(result["parse_ok"])
        self.assertEqual(result["tasks"], [])


class TestPromptTemplates(unittest.TestCase):
    def test_all_three_templates_exist_as_files(self) -> None:
        for name in ("planner.md", "critic.md", "writer.md"):
            self.assertTrue((llm.PROMPTS_DIR / name).is_file(), name)

    def _rendered_planner(self) -> str:
        from argus.orchestrator import tools

        return llm.render_prompt(
            "planner.md",
            {
                "goal": "check t for changes",
                "target_id": "t",
                "tool_catalog": tools.TOOL_CATALOG,
                "max_tasks": 5,
            },
        )

    def test_planner_prompt_names_every_schema_field(self) -> None:
        """Guards the drift that caused a 5/5 first-attempt failure rate: the
        schema required `tool_name` and `done`, but the prompt never spelled
        either, so the model reached for a plausible shape (`tool`, and `done`
        omitted) until the retry fed the error back.
        """
        rendered = self._rendered_planner()
        schema = llm.PlannerOutput.model_json_schema()

        fields = set(schema["properties"]) | set(schema["$defs"]["Action"]["properties"])
        missing = sorted(f for f in fields if f not in rendered)
        self.assertEqual(missing, [], f"schema fields never named in planner.md: {missing}")

    def test_planner_prompt_example_validates_against_the_schema(self) -> None:
        """A wrong inline example would teach the wrong shape more strongly
        than the prose corrects it, so the example itself must parse.
        """
        rendered = self._rendered_planner()

        start = rendered.index('{\n  "thought"')
        depth, end = 0, None
        for offset, char in enumerate(rendered[start:], start):
            depth += (char == "{") - (char == "}")
            if depth == 0:
                end = offset + 1
                break
        self.assertIsNotNone(end, "could not find a balanced JSON example in planner.md")

        parsed = llm.PlannerOutput.model_validate_json(rendered[start:end])
        self.assertEqual(parsed.tasks[0].tool_name, "stub_fetch_ok")
        self.assertIs(parsed.done, False)

    def test_rendering_substitutes_and_leaves_no_placeholders(self) -> None:
        rendered = llm.render_prompt(
            "planner.md",
            {"goal": "GOAL-X", "target_id": "TARGET-X", "tool_catalog": "CATALOG-X", "max_tasks": 5},
        )
        self.assertIn("GOAL-X", rendered)
        self.assertIn("CATALOG-X", rendered)
        self.assertNotIn("$goal", rendered)
        self.assertNotIn("$tool_catalog", rendered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
