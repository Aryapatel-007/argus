"""Embedding module tests.

Run: python -m unittest discover -s tests -v

UNLIKE test_llm.py, most of these hit a REAL local Ollama. The whole point of
this module is what a real model actually returns — width, determinism,
relative ordering, and what an empty string does — and a mock would only ever
confirm what the mock was told to say. Tests that need a specific failure the
real server will not produce on demand (a wrong-width vector, a timeout) fake
the response at the client boundary and say so.

If Ollama is not running, the real-call tests SKIP rather than fail: a missing
server is an environment fact, not a defect in this module.
"""

from __future__ import annotations

import asyncio
import math
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import List
from unittest import mock

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from argus import config, embeddings  # noqa: E402
from argus.orchestrator import llm  # noqa: E402
from argus.storage import db  # noqa: E402
from argus.storage.migrate import apply_migrations  # noqa: E402


class TestLayering(unittest.TestCase):
    """embeddings.py must depend on the neutral module, never on the
    orchestrator. A subprocess so the import graph is checked from a genuinely
    fresh interpreter, not one where orchestrator modules are already loaded
    by test collection itself.
    """

    def test_importing_embeddings_does_not_pull_in_orchestrator(self) -> None:
        code = (
            "import sys; sys.path.insert(0, %r);"
            "import argus.embeddings;"
            "assert not any(m.startswith('argus.orchestrator') for m in sys.modules), "
            "sorted(m for m in sys.modules if m.startswith('argus.orchestrator'));"
            "print('OK')" % str(SRC)
        )
        import subprocess

        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("OK", result.stdout)

# Three short fixed strings. The first two are about the same thing, the third
# is not. Assertions are on ORDERING only — absolute cosine values from
# nomic-embed-text sit in a narrow band and are not a stable thing to assert.
SIMILAR_A = "The Master of Artificial Intelligence has February and July intakes."
SIMILAR_B = "This AI masters degree starts in February or July each year."
UNRELATED = "Sourdough bread needs a starter and a long cold proof."


def cosine(u: List[float], v: List[float]) -> float:
    du = math.sqrt(sum(x * x for x in u))
    dv = math.sqrt(sum(x * x for x in v))
    return sum(x * y for x, y in zip(u, v)) / (du * dv)


def ollama_available() -> bool:
    try:
        from ollama import Client

        Client(host=config.ollama_options().get("host"), timeout=5).ps()
        return True
    except Exception:
        return False


OLLAMA_UP = ollama_available()
requires_ollama = unittest.skipUnless(
    OLLAMA_UP, "local Ollama not reachable; real-call tests skipped"
)


class FakeEmbedResponse:
    """Mirrors the field set of a real ollama EmbedResponse.

    Values taken from a real nomic-embed-text call: prompt_eval_count is
    populated, eval_count and eval_duration are genuinely absent.
    """

    def __init__(self, embeddings_, prompt_eval_count=9, total_duration=4302553600):
        self.embeddings = embeddings_
        self.prompt_eval_count = prompt_eval_count
        self.eval_count = None
        self.eval_duration = None
        self.total_duration = total_duration
        self.load_duration = 2817178800


class EmbeddingsTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "embed.db"
        self.addCleanup(self._tmp.cleanup)
        apply_migrations(self.db_path)
        with db.transaction(self.db_path) as conn:
            conn.execute(
                "INSERT INTO runs (run_id, target_id, status) VALUES (?, ?, ?)",
                ("run-1", "monash_ai_masters", "running"),
            )

    async def asyncSetUp(self) -> None:
        # IsolatedAsyncioTestCase turns asyncio debug mode on, which logs any
        # callback over 0.1s as suspiciously slow. A real embed call awaits the
        # network for ~0.1-0.2s and trips it every time. Raising the threshold
        # keeps that noise out of the output so a genuine stall still stands
        # out; it does not disable the check.
        asyncio.get_running_loop().slow_callback_duration = 5.0

        # Each test gets its own event loop, so each gets its own client. Close
        # it on the way out rather than leaving the pool to the garbage
        # collector, which on Windows surfaces as ResourceWarning noise.
        self.addAsyncCleanup(llm.close_async_clients)

    def rows(self):
        with db.connect(self.db_path) as conn:
            return conn.execute("SELECT * FROM llm_calls").fetchall()


class TestConfigWiring(EmbeddingsTestCase):
    def test_model_tag_comes_from_config_and_is_not_the_chat_model(self) -> None:
        """A hardcoded tag, or accidentally reusing the chat model, both fail here."""
        tag = embeddings.model_tag()
        self.assertEqual(tag, config.load_config()["models"]["embed"])
        self.assertNotEqual(tag, config.model_for_role("planner"))

    def test_dimension_constant_matches_the_facts_vec_column(self) -> None:
        """One source of truth for 768, verified rather than duplicated.

        The schema is the authority — Python cannot import a .sql file — so the
        constant is checked against the width the live database actually
        declares. If 002 and this module ever disagree, this fails.
        """
        with db.connect(self.db_path) as conn:
            sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'facts_vec'"
            ).fetchone()[0]

        self.assertIn(f"FLOAT[{embeddings.EMBEDDING_DIMENSIONS}]", sql)


@requires_ollama
class TestRealEmbedding(EmbeddingsTestCase):
    async def test_returns_768_finite_floats(self) -> None:
        vector = await embeddings.embed(SIMILAR_A, run_id="run-1")

        self.assertEqual(len(vector), 768)
        self.assertTrue(all(isinstance(x, float) for x in vector))
        self.assertTrue(all(math.isfinite(x) for x in vector))

    async def test_same_input_twice_is_identical(self) -> None:
        """nomic-embed-text is deterministic — MEASURED, not assumed.

        This matters beyond tidiness: change detection compares content hashes
        first, but any future vector-level comparison depends on the same text
        producing the same vector. If this ever fails, change-detection design
        has to change with it, so the assertion is exact equality rather than
        a tolerance.
        """
        first = await embeddings.embed(SIMILAR_A, run_id="run-1")
        second = await embeddings.embed(SIMILAR_A, run_id="run-1")

        self.assertEqual(first, second)

    async def test_different_strings_give_different_vectors(self) -> None:
        a = await embeddings.embed(SIMILAR_A, run_id="run-1")
        b = await embeddings.embed(UNRELATED, run_id="run-1")

        self.assertNotEqual(a, b)

    async def test_similar_pair_outranks_unrelated_pair(self) -> None:
        """Ordering only, never absolute values.

        The measured margin is thin (~0.02 on this model), but embeddings are
        deterministic so this is reproducible rather than flaky. See
        notes/sprint-2.md — the narrow band is a recorded finding, not
        something this test papers over.
        """
        a = await embeddings.embed(SIMILAR_A, run_id="run-1")
        b = await embeddings.embed(SIMILAR_B, run_id="run-1")
        c = await embeddings.embed(UNRELATED, run_id="run-1")

        self.assertGreater(cosine(a, b), cosine(a, c))

    async def test_list_input_returns_a_list_of_vectors(self) -> None:
        vectors = await embeddings.embed([SIMILAR_A, UNRELATED], run_id="run-1")

        self.assertEqual(len(vectors), 2)
        self.assertTrue(all(len(v) == 768 for v in vectors))
        self.assertNotEqual(vectors[0], vectors[1])

    async def test_single_string_does_not_return_a_nested_list(self) -> None:
        """Shape mirrors the input, so one fact never needs unwrapping."""
        vector = await embeddings.embed(SIMILAR_A, run_id="run-1")

        self.assertIsInstance(vector[0], float)


@requires_ollama
class TestEmptyStringNegativeControl(EmbeddingsTestCase):
    """NEGATIVE CONTROL: what does the model do with nothing?

    MEASURED BEHAVIOUR: an empty string is NOT rejected and does NOT come back
    as zeros. nomic-embed-text returns a perfectly ordinary unit vector
    (norm 1.0, non-zero components), and a whitespace-only string returns the
    IDENTICAL vector.

    That is the finding, and it is a trap rather than a convenience: an empty
    or blank fact will embed successfully, land in facts_vec, and match
    queries like any other row. Nothing in this layer can distinguish it. The
    memory server must reject empty content BEFORE embedding — this module
    deliberately does not, because silently rewriting a caller's input is
    worse than returning what the model actually said.
    """

    async def test_empty_string_returns_a_normal_vector_not_zeros_and_not_an_error(
        self,
    ) -> None:
        vector = await embeddings.embed("", run_id="run-1")

        self.assertEqual(len(vector), 768)
        self.assertFalse(all(x == 0.0 for x in vector), "expected a real vector, not zeros")
        norm = math.sqrt(sum(x * x for x in vector))
        self.assertAlmostEqual(norm, 1.0, places=4)

    async def test_whitespace_only_matches_empty_string_exactly(self) -> None:
        blank = await embeddings.embed("", run_id="run-1")
        spaces = await embeddings.embed("   ", run_id="run-1")

        self.assertEqual(blank, spaces)

    async def test_empty_string_is_indistinguishable_from_real_content_downstream(
        self,
    ) -> None:
        """The consequence, asserted so it cannot be forgotten.

        An empty fact produces a storable, queryable vector. If a later change
        makes empty input raise instead, this test fails and forces the
        decision to be made deliberately rather than absorbed.
        """
        vector = await embeddings.embed("", run_id="run-1")
        blob = embeddings.serialize(vector)

        with db.transaction(self.db_path) as conn:
            conn.execute(
                "INSERT INTO facts_vec (fact_id, embedding) VALUES (?, ?)", (1, blob)
            )

        with db.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT fact_id FROM facts_vec WHERE embedding MATCH ? "
                "ORDER BY distance LIMIT 1",
                (blob,),
            ).fetchall()

        self.assertEqual([r["fact_id"] for r in rows], [1])


class TestSerialize(EmbeddingsTestCase):
    def test_produces_a_3072_byte_blob(self) -> None:
        blob = embeddings.serialize([0.5] * 768)

        self.assertIsInstance(blob, bytes)
        self.assertEqual(len(blob), 3072)
        self.assertEqual(len(blob), embeddings.SERIALIZED_BYTES)

    def test_wrong_width_raises_the_named_exception(self) -> None:
        with self.assertRaises(embeddings.EmbeddingDimensionError):
            embeddings.serialize([0.5] * 767)

    def test_round_trips_through_facts_vec_and_a_knn_query(self) -> None:
        """The real reason this helper exists: what goes in must come back out."""
        vectors = [[float(i) / 1000] * 768 for i in (1, 500, 900)]
        with db.transaction(self.db_path) as conn:
            for fact_id, vector in enumerate(vectors, start=1):
                conn.execute(
                    "INSERT INTO facts_vec (fact_id, embedding) VALUES (?, ?)",
                    (fact_id, embeddings.serialize(vector)),
                )

        with db.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT fact_id, distance FROM facts_vec "
                "WHERE embedding MATCH ? ORDER BY distance LIMIT 3",
                (embeddings.serialize(vectors[0]),),
            ).fetchall()

        self.assertEqual(len(rows), 3)
        # Querying with vector 1 must rank vector 1 first, at distance ~0.
        self.assertEqual(rows[0]["fact_id"], 1)
        self.assertAlmostEqual(rows[0]["distance"], 0.0, places=4)


@requires_ollama
class TestLlmCallsLogging(EmbeddingsTestCase):
    async def test_row_lands_with_embedder_role_and_null_output_metrics(self) -> None:
        """NULL, not 0 — Ollama omits these for embeddings entirely.

        A 0 here would assert a measurement nobody took and would drag any
        future AVG(tok_s) toward zero.
        """
        with db.connect(self.db_path) as conn:
            await embeddings.embed(SIMILAR_A, run_id="run-1", conn=conn)

        rows = self.rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]

        self.assertEqual(row["role"], "embedder")
        self.assertEqual(row["model_tag"], embeddings.model_tag())
        self.assertEqual(row["think_mode"], "n/a")
        self.assertEqual(row["ok"], 1)
        self.assertEqual(row["timed_out"], 0)
        self.assertEqual(row["run_id"], "run-1")

        for column in ("eval_count", "eval_duration_ns", "tok_s"):
            with self.subTest(column=column):
                self.assertIsNone(row[column])

        for column in ("task_id", "step_index", "prompt_file", "raw_response"):
            with self.subTest(column=column):
                self.assertIsNone(row[column])

        self.assertIsNotNone(row["prompt_eval_count"])
        self.assertIsNotNone(row["total_duration_ns"])

    async def test_task_id_and_step_index_are_recorded_when_supplied(self) -> None:
        with db.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO tasks (task_id, run_id, task_index, description, status) "
                "VALUES (?, ?, ?, ?, ?)",
                ("task-1", "run-1", 0, "check intake dates", "pending"),
            )
            await embeddings.embed(
                SIMILAR_A, run_id="run-1", conn=conn, task_id="task-1", step_index=4
            )

        row = self.rows()[0]
        self.assertEqual(row["task_id"], "task-1")
        self.assertEqual(row["step_index"], 4)

    async def test_no_connection_means_no_row_and_no_crash(self) -> None:
        """The memory server may embed before it has opened a database."""
        vector = await embeddings.embed(SIMILAR_A, run_id="run-1")

        self.assertEqual(len(vector), 768)
        self.assertEqual(self.rows(), [])


class TestFakedResponses(EmbeddingsTestCase):
    """Failures the real server will not produce on demand.

    Faked at `embeddings._invoke`, the module's single network seam — the same
    place test_llm.py patches for the chat path.
    """

    @staticmethod
    def _returning(response):
        async def fake(tag, inputs):
            return response

        return fake

    @staticmethod
    def _raising(exc):
        async def fake(tag, inputs):
            raise exc

        return fake

    async def test_wrong_dimension_raises_the_named_exception(self) -> None:
        """Cannot happen today. Will happen the day models.embed changes.

        A bare AssertionError would be wrong twice over: it vanishes under
        `python -O`, and it reads as an internal invariant rather than the
        model-boundary check this is.
        """
        with mock.patch.object(
            embeddings, "_invoke", self._returning(FakeEmbedResponse([[0.1] * 512]))
        ):
            with db.connect(self.db_path) as conn:
                with self.assertRaises(embeddings.EmbeddingDimensionError) as caught:
                    await embeddings.embed("anything", run_id="run-1", conn=conn)

        self.assertEqual(caught.exception.expected, 768)
        self.assertEqual(caught.exception.actual, 512)
        self.assertIn("512", str(caught.exception))

    async def test_wrong_dimension_still_writes_its_llm_calls_row(self) -> None:
        """A call that happened and failed is data. Never skip the insert."""
        with mock.patch.object(
            embeddings, "_invoke", self._returning(FakeEmbedResponse([[0.1] * 512]))
        ):
            with db.connect(self.db_path) as conn:
                with self.assertRaises(embeddings.EmbeddingDimensionError):
                    await embeddings.embed("anything", run_id="run-1", conn=conn)

        row = self.rows()[0]
        self.assertEqual(row["role"], "embedder")
        self.assertEqual(row["ok"], 0)
        self.assertEqual(row["timed_out"], 0)
        self.assertIn("512", row["error"])

    async def test_timeout_sets_timed_out_and_still_writes_the_row(self) -> None:
        import httpx

        with mock.patch.object(
            embeddings, "_invoke", self._raising(httpx.ReadTimeout("too slow"))
        ):
            with db.connect(self.db_path) as conn:
                with self.assertRaises(embeddings.EmbeddingCallError) as caught:
                    await embeddings.embed("anything", run_id="run-1", conn=conn)

        self.assertTrue(caught.exception.timed_out)

        row = self.rows()[0]
        self.assertEqual(row["role"], "embedder")
        self.assertEqual(row["ok"], 0)
        self.assertEqual(row["timed_out"], 1)
        self.assertIn("timeout", row["error"].lower())
        self.assertIsNone(row["eval_count"])
        self.assertIsNone(row["tok_s"])

    async def test_unreachable_server_writes_a_row_that_is_not_a_timeout(self) -> None:
        with mock.patch.object(
            embeddings, "_invoke", self._raising(ConnectionError("refused"))
        ):
            with db.connect(self.db_path) as conn:
                with self.assertRaises(embeddings.EmbeddingCallError):
                    await embeddings.embed("anything", run_id="run-1", conn=conn)

        row = self.rows()[0]
        self.assertEqual(row["ok"], 0)
        self.assertEqual(row["timed_out"], 0)
        self.assertIn("unreachable", row["error"])

    async def test_missing_vector_for_a_single_input_raises_count_error(self) -> None:
        """Guards an IndexError that would otherwise surface far from its cause."""
        with mock.patch.object(
            embeddings, "_invoke", self._returning(FakeEmbedResponse([]))
        ):
            with self.assertRaises(embeddings.EmbeddingCountError):
                await embeddings.embed("anything", run_id="run-1")


class TestAsyncClientIsCached(EmbeddingsTestCase):
    """Sprint 1 fixed a leaked connection pool. Do not reintroduce it."""

    async def test_repeated_calls_reuse_one_client(self) -> None:
        llm.reset_async_clients()
        self.addCleanup(llm.reset_async_clients)

        host = config.ollama_options().get("host")
        first = llm.get_async_client(host, llm.TIMEOUT_SECONDS)
        second = llm.get_async_client(host, llm.TIMEOUT_SECONDS)

        self.assertIs(first, second)

    async def test_sync_and_async_clients_are_separate_objects(self) -> None:
        """They cannot share a pool — httpx.Client and httpx.AsyncClient differ."""
        llm.reset_async_clients()
        llm._get_client.cache_clear()
        self.addCleanup(llm.reset_async_clients)
        self.addCleanup(llm._get_client.cache_clear)

        host = config.ollama_options().get("host")
        self.assertIsNot(
            llm.get_async_client(host, llm.TIMEOUT_SECONDS),
            llm._get_client(host, llm.TIMEOUT_SECONDS),
        )

    async def test_a_second_event_loop_does_not_inherit_a_dead_client(self) -> None:
        """Regression test for the bug this cache design was written to avoid.

        An httpx.AsyncClient binds its pool to the loop that built it. Caching
        on (host, timeout) alone handed a closed loop's client to the next one
        and every call after the first failed with "Event loop is closed".
        Caught by the test suite, which runs one loop per test; production runs
        a single long-lived loop and would have hidden it indefinitely.

        The second loop runs in a thread because asyncio.run() cannot be called
        from inside a running loop.
        """
        host = config.ollama_options().get("host")
        this_loop_client = llm.get_async_client(host, llm.TIMEOUT_SECONDS)

        captured = {}

        async def in_another_loop():
            captured["client"] = llm.get_async_client(host, llm.TIMEOUT_SECONDS)
            await llm.close_async_clients()

        thread = threading.Thread(target=lambda: asyncio.run(in_another_loop()))
        thread.start()
        thread.join(timeout=30)

        self.assertFalse(thread.is_alive(), "second-loop thread hung")
        self.assertIsNot(captured["client"], this_loop_client)
        # ...and this loop's client is still the one it started with.
        self.assertIs(
            llm.get_async_client(host, llm.TIMEOUT_SECONDS), this_loop_client
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
