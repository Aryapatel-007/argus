"""Memory MCP server tests.

Run: python -m unittest discover -s tests -v

Uses the v2 in-memory `Client(mcp)` pattern — no subprocess, no Node, no MCP
Inspector. That is the point of choosing v2 (design-decisions.md): Inspector
becomes confirmation rather than the only way to exercise a tool.

Embedding calls are REAL by default and skip if Ollama is unreachable, since
the whole embed-once/skip-on-unchanged behaviour is what several of these
tests exist to pin down.
"""

from __future__ import annotations

import asyncio
import io
import math
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from mcp import Client  # noqa: E402

from argus import config, embeddings  # noqa: E402
from argus.orchestrator import llm  # noqa: E402
from argus.servers import memory, scoring  # noqa: E402
from argus.storage import db  # noqa: E402
from argus.storage.migrate import apply_migrations  # noqa: E402

RUN_ID = "run-memory-1"
TARGET = "monash_ai_masters"
ENTITY = "intake_dates"

CONTENT_A = "Intake dates: February and July."
CONTENT_B = "Intake dates: February, July and November."


def ollama_available() -> bool:
    try:
        from ollama import Client as OllamaClient

        OllamaClient(host=config.ollama_options().get("host"), timeout=5).ps()
        return True
    except Exception:
        return False


OLLAMA_UP = ollama_available()
requires_ollama = unittest.skipUnless(
    OLLAMA_UP, "local Ollama not reachable; memory-server tests skipped"
)


class MemoryServerTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "memory.db"
        self.addCleanup(self._tmp.cleanup)
        apply_migrations(self.db_path)
        with db.transaction(self.db_path) as conn:
            conn.execute(
                "INSERT INTO runs (run_id, target_id, status) VALUES (?, ?, ?)",
                (RUN_ID, TARGET, "running"),
            )
        self.mcp = memory.build_server(self.db_path, RUN_ID)

    async def asyncSetUp(self) -> None:
        asyncio.get_running_loop().slow_callback_duration = 10.0
        self.addAsyncCleanup(llm.close_async_clients)

    async def remember(self, client, **kwargs):
        payload = {
            "target_id": TARGET,
            "entity_key": ENTITY,
            "content": CONTENT_A,
        }
        payload.update(kwargs)
        return await client.call_tool("remember_fact", payload)

    # -- introspection helpers ---------------------------------------------

    def facts(self):
        with db.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT * FROM facts ORDER BY fact_id"
            ).fetchall()

    def vec_ids(self):
        with db.connect(self.db_path) as conn:
            return [r["fact_id"] for r in conn.execute(
                "SELECT fact_id FROM facts_vec ORDER BY fact_id"
            )]

    def embed_call_count(self) -> int:
        with db.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT COUNT(*) AS n FROM llm_calls WHERE role = 'embedder'"
            ).fetchone()["n"]


@requires_ollama
class TestRememberFactAdded(MemoryServerTestCase):
    async def test_new_entity_is_added_at_version_1(self) -> None:
        async with Client(self.mcp) as client:
            result = await self.remember(client)

        self.assertFalse(result.is_error, result.content)
        payload = result.structured_content
        self.assertEqual(payload["change_kind"], "added")
        self.assertEqual(payload["version"], 1)
        self.assertIsNone(payload["previous_fact_id"])

        rows = self.facts()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["is_current"], 1)
        self.assertEqual(rows[0]["change_kind"], "added")
        self.assertEqual(rows[0]["first_seen_run_id"], RUN_ID)

        # The vector landed too, keyed by the same fact_id.
        self.assertEqual(self.vec_ids(), [rows[0]["fact_id"]])
        self.assertEqual(self.embed_call_count(), 1)


@requires_ollama
class TestRememberFactUnchanged(MemoryServerTestCase):
    async def test_identical_content_does_not_insert_or_re_embed(self) -> None:
        """The expensive half of change detection: an unchanged page must cost
        zero Ollama time on the second run.
        """
        async with Client(self.mcp) as client:
            first = await self.remember(client)
            embeds_after_first = self.embed_call_count()
            second = await self.remember(client)

        self.assertEqual(second.structured_content["change_kind"], "unchanged")
        self.assertEqual(second.structured_content["version"], 1)
        self.assertEqual(
            second.structured_content["fact_id"],
            first.structured_content["fact_id"],
        )

        # ONE row, not two.
        rows = self.facts()
        self.assertEqual(len(rows), 1)

        # facts_vec untouched, and — the real assertion — NO second embed call.
        # Checking llm_calls rather than the vector table is deliberate: an
        # implementation that embedded and then discarded the vector would
        # leave facts_vec looking correct while still paying the ~0.9s.
        self.assertEqual(self.vec_ids(), [rows[0]["fact_id"]])
        self.assertEqual(self.embed_call_count(), embeds_after_first)
        self.assertEqual(self.embed_call_count(), 1)

    async def test_unchanged_still_bumps_last_seen(self) -> None:
        async with Client(self.mcp) as client:
            await self.remember(client)
            before = self.facts()[0]
            await asyncio.sleep(0.01)
            await self.remember(client)

        after = self.facts()[0]
        self.assertEqual(after["fact_id"], before["fact_id"])
        self.assertGreaterEqual(after["last_seen_at"], before["last_seen_at"])
        self.assertEqual(after["last_seen_run_id"], RUN_ID)
        # first_seen must NOT move — the entity was discovered on run one.
        self.assertEqual(after["first_seen_at"], before["first_seen_at"])


@requires_ollama
class TestRememberFactModified(MemoryServerTestCase):
    async def test_changed_content_versions_and_preserves_history(self) -> None:
        async with Client(self.mcp) as client:
            first = await self.remember(client)
            second = await self.remember(client, content=CONTENT_B)

        old_id = first.structured_content["fact_id"]
        new_id = second.structured_content["fact_id"]

        self.assertEqual(second.structured_content["change_kind"], "modified")
        self.assertEqual(second.structured_content["version"], 2)
        self.assertEqual(second.structured_content["previous_fact_id"], old_id)

        rows = {r["fact_id"]: r for r in self.facts()}
        self.assertEqual(len(rows), 2, "history must be preserved, not overwritten")

        self.assertEqual(rows[old_id]["is_current"], 0)
        self.assertEqual(rows[new_id]["is_current"], 1)
        self.assertEqual(rows[new_id]["supersedes_id"], old_id)
        self.assertEqual(rows[new_id]["version"], 2)
        self.assertEqual(rows[new_id]["change_kind"], "modified")

        # first_seen carries forward: the ENTITY was first seen at version 1.
        self.assertEqual(
            rows[new_id]["first_seen_at"], rows[old_id]["first_seen_at"]
        )

        # Both embeddings present — facts_vec is keyed by fact_id, so the old
        # version's vector is not overwritten by the new one.
        self.assertEqual(sorted(self.vec_ids()), sorted([old_id, new_id]))
        self.assertEqual(self.embed_call_count(), 2)

    async def test_only_one_current_row_survives_a_modification(self) -> None:
        async with Client(self.mcp) as client:
            await self.remember(client)
            await self.remember(client, content=CONTENT_B)

        with db.connect(self.db_path) as conn:
            current = conn.execute(
                "SELECT COUNT(*) AS n FROM facts "
                "WHERE target_id = ? AND entity_key = ? AND is_current = 1",
                (TARGET, ENTITY),
            ).fetchone()["n"]
        self.assertEqual(current, 1)


class TestEmptyContentRejected(MemoryServerTestCase):
    """No Ollama needed: rejection must happen before any embed call."""

    async def test_empty_string_is_rejected_with_a_readable_error(self) -> None:
        async with Client(self.mcp) as client:
            result = await self.remember(client, content="")

        self.assertTrue(result.is_error)
        message = result.content[0].text
        self.assertIn("empty", message.lower())

        self.assertEqual(self.facts(), [])
        self.assertEqual(self.vec_ids(), [])
        self.assertEqual(self.embed_call_count(), 0)

    async def test_whitespace_only_is_rejected_the_same_way(self) -> None:
        async with Client(self.mcp) as client:
            result = await self.remember(client, content="   \n\t  ")

        self.assertTrue(result.is_error)
        self.assertIn("empty", result.content[0].text.lower())

        self.assertEqual(self.facts(), [])
        self.assertEqual(self.vec_ids(), [])
        self.assertEqual(self.embed_call_count(), 0)


@requires_ollama
class TestConcurrentRememberFact(MemoryServerTestCase):
    """The partial unique index under REAL contention, not migration-time.

    Plain asyncio.gather is NOT sufficient proof of overlap on its own: MCP v2
    could in principle dispatch tool calls one after another, and two calls
    that never actually interleave inside the tool body would make every
    assertion below true for a boring, uninteresting reason. Every test here
    therefore forces and PROVES genuine overlap with an asyncio.Event barrier
    that blocks every racing call at the same await point
    (embeddings.embed, the tool's only real yield) until all of them have
    arrived — the same technique that measured the 0.198s overlap this
    server's per-call-connection design was built around. If the barrier does
    not open (not every call reached it), the test fails loudly before ever
    reaching its real assertions, rather than silently passing on a race that
    didn't happen.
    """

    async def race(self, client, contents: list) -> list:
        """Fire len(contents) concurrent remember_fact calls for the SAME
        (TARGET, ENTITY), each with a different content string, and return
        their CallToolResults once all have completed.

        Patches embeddings.embed at module level for the duration of the race
        only, and restores it unconditionally in `finally` even if a call
        raises through gather — the barrier must never leak into a later test.
        """
        n = len(contents)
        arrived = 0
        all_arrived = asyncio.Event()
        real_embed = embeddings.embed

        async def barrier_embed(*args, **kwargs):
            nonlocal arrived
            arrived += 1
            if arrived == n:
                all_arrived.set()
            await all_arrived.wait()
            return await real_embed(*args, **kwargs)

        embeddings.embed = barrier_embed
        try:
            results = await asyncio.gather(
                *(self.remember(client, content=c) for c in contents),
                return_exceptions=True,
            )
        finally:
            embeddings.embed = real_embed

        self.assertEqual(
            arrived, n,
            f"only {arrived}/{n} calls reached the embed barrier — they did not "
            f"genuinely overlap, so this race is not a real test of anything",
        )
        for result in results:
            self.assertNotIsInstance(
                result, BaseException, f"call raised through gather: {result!r}"
            )
        return results

    async def test_two_concurrent_calls_leave_exactly_one_current_row(self) -> None:
        """Confirmed real behaviour, not assumed: under this server's design
        (single asyncio event loop, no further await inside db.atomic()), the
        double-checked lookup means the SECOND call to reach BEGIN IMMEDIATE
        always observes the FIRST call's already-committed row, because the
        first call runs synchronously through to COMMIT before yielding
        control back to the loop. There is no window in which both calls can
        reach the INSERT with a stale "no current row" view at once.

        The consequence: the loser never hits the partial unique index at all.
        It does not raise IntegrityError, and it does not need a retry loop -
        it transitions cleanly to 'modified', with previous_fact_id pointing at
        the winner's row. This was verified empirically (3 real runs against a
        real Ollama server, winner order varied, outcome shape did not) before
        writing this assertion, specifically so it states what was OBSERVED
        rather than what the design merely intends.

        Because the outcome is a normal, non-error 'modified' result, this is
        NOT a gap the caller needs to handle specially - remember_fact already
        returns something the planner can read and act on. No follow-up item.
        """
        async with Client(self.mcp) as client:
            results = await self.race(client, [CONTENT_A, CONTENT_B])

        for result in results:
            self.assertFalse(
                result.is_error, f"a racing call failed: {result.content}"
            )

        kinds = sorted(r.structured_content["change_kind"] for r in results)
        self.assertEqual(
            kinds, ["added", "modified"],
            "the loser must resolve to a clean 'modified' write, not an error "
            "and not a silently dropped call",
        )

        winner = next(r for r in results if r.structured_content["change_kind"] == "added")
        loser = next(r for r in results if r.structured_content["change_kind"] == "modified")
        self.assertEqual(
            loser.structured_content["previous_fact_id"],
            winner.structured_content["fact_id"],
            "the loser's write must chain onto the winner's row, not float free",
        )
        self.assertEqual(loser.structured_content["version"], 2)
        self.assertEqual(winner.structured_content["version"], 1)

        with db.connect(self.db_path) as conn:
            current = conn.execute(
                "SELECT COUNT(*) AS n FROM facts "
                "WHERE target_id = ? AND entity_key = ? AND is_current = 1",
                (TARGET, ENTITY),
            ).fetchone()["n"]
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]

        self.assertEqual(current, 1, "the partial unique index invariant broke")
        self.assertEqual(integrity, "ok")

        # Whatever order they landed in, versions must be a clean 1..N chain
        # with no gaps and no duplicates — that is what proves neither call
        # wrote a partial row.
        versions = sorted(r["version"] for r in self.facts())
        self.assertEqual(versions, list(range(1, len(versions) + 1)))

    async def test_concurrent_identical_content_does_not_double_insert(self) -> None:
        async with Client(self.mcp) as client:
            results = await self.race(client, [CONTENT_A, CONTENT_A])

        for result in results:
            self.assertFalse(
                result.is_error, f"a racing call failed: {result.content}"
            )

        kinds = sorted(r.structured_content["change_kind"] for r in results)
        self.assertEqual(
            kinds, ["added", "unchanged"],
            "one call must add and the other must resolve to unchanged",
        )
        self.assertEqual(len(self.facts()), 1)

    async def test_three_way_race_still_leaves_exactly_one_current_row(self) -> None:
        """Two racers is the minimum case; three is what actually rules out an
        off-by-one in the double-checked pattern (e.g. a lock held for only
        the FIRST contender, not every subsequent one).
        """
        async with Client(self.mcp) as client:
            results = await self.race(client, [CONTENT_A, CONTENT_B, "version C"])

        for result in results:
            self.assertFalse(
                result.is_error, f"a racing call failed: {result.content}"
            )

        kinds = sorted(r.structured_content["change_kind"] for r in results)
        self.assertEqual(kinds, ["added", "modified", "modified"])

        with db.connect(self.db_path) as conn:
            current = conn.execute(
                "SELECT COUNT(*) AS n FROM facts "
                "WHERE target_id = ? AND entity_key = ? AND is_current = 1",
                (TARGET, ENTITY),
            ).fetchone()["n"]
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]

        self.assertEqual(current, 1)
        self.assertEqual(integrity, "ok")
        versions = sorted(r["version"] for r in self.facts())
        self.assertEqual(versions, [1, 2, 3])


class TestPerCallConnectionOverhead(MemoryServerTestCase):
    """remember_fact opens a fresh connection every call (see its docstring on
    why a shared one cannot work under concurrency). Confirms that connection
    still goes through db.connect()'s FULL setup — WAL, busy_timeout,
    sqlite_vec.load() — rather than some lighter-weight path a future
    "optimize the per-call pattern" change might be tempted to introduce.

    Measured cost: 200 consecutive db.connect() opens against a migrated
    database averaged 1.919ms each (0.3838s / 200). Negligible against the
    ~0.9s an embed call costs, so there is no performance reason to ever
    special-case this — a correctness argument (needing separate transactions
    per concurrent caller) already settled it, but it's worth knowing the
    "just share a connection to save time" temptation would have bought
    nothing anyway.
    """

    def test_per_call_connection_carries_the_full_setup(self) -> None:
        with db.connect(self.db_path) as conn:
            self.assertEqual(conn.execute("PRAGMA journal_mode").fetchone()[0].lower(), "wal")
            self.assertEqual(conn.execute("PRAGMA busy_timeout").fetchone()[0], 5000)
            self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            # Only succeeds if sqlite_vec.load() actually ran on THIS connection.
            self.assertTrue(
                conn.execute("SELECT vec_version()").fetchone()[0].startswith("v0.1")
            )


class TestScoring(unittest.TestCase):
    """Pure scorer, no database and no Ollama. Ranking logic must be testable
    without either, since a bad ordering is far easier to diagnose here than
    through a full recall.
    """

    def test_relevance_direction_and_bounds(self) -> None:
        """Closer must score higher, and the range must be exactly [0, 1]."""
        self.assertEqual(scoring.relevance_from_l2(0.0), 1.0)
        self.assertEqual(scoring.relevance_from_l2(scoring.MAX_UNIT_L2), 0.0)
        self.assertGreater(
            scoring.relevance_from_l2(0.5), scoring.relevance_from_l2(1.5)
        )

    def test_relevance_matches_the_cosine_identity(self) -> None:
        """1 - L2^2/4 must equal (cos + 1)/2 for unit vectors.

        This is the whole justification for the conversion — if it drifts, the
        relevance number stops meaning "rescaled cosine similarity" and becomes
        an arbitrary squash.
        """
        for cos in (-1.0, -0.5, 0.0, 0.42, 0.756, 1.0):
            with self.subTest(cos=cos):
                l2 = math.sqrt(max(0.0, 2 - 2 * cos))
                self.assertAlmostEqual(
                    scoring.relevance_from_l2(l2), (cos + 1) / 2, places=9
                )

    def test_negative_distance_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            scoring.relevance_from_l2(-0.1)

    def test_recency_decays_with_age(self) -> None:
        now = datetime(2026, 8, 24, tzinfo=timezone.utc)
        fresh = scoring.recency_from_last_seen("2026-08-24T00:00:00.000Z", now=now)
        half = scoring.recency_from_last_seen("2026-07-25T00:00:00.000Z", now=now)
        old = scoring.recency_from_last_seen("2026-02-24T00:00:00.000Z", now=now)

        self.assertAlmostEqual(fresh, 1.0, places=6)
        self.assertAlmostEqual(half, 0.5, places=2)  # ~30 days = one half-life
        self.assertLess(old, half)
        self.assertGreater(old, 0.0, "exponential decay must never reach zero")

    def test_future_timestamp_clamps_to_one(self) -> None:
        """Clock skew must not produce a recency above 1.0 and break the blend."""
        now = datetime(2026, 8, 24, tzinfo=timezone.utc)
        self.assertEqual(
            scoring.recency_from_last_seen("2027-01-01T00:00:00.000Z", now=now), 1.0
        )

    def test_unparseable_timestamp_sorts_last_rather_than_raising(self) -> None:
        self.assertEqual(scoring.recency_from_last_seen("not a date"), 0.0)

    def test_weights_are_the_agreed_split(self) -> None:
        self.assertEqual(scoring.RELEVANCE_WEIGHT, 0.7)
        self.assertEqual(scoring.RECENCY_WEIGHT, 0.3)
        self.assertAlmostEqual(
            scoring.RELEVANCE_WEIGHT + scoring.RECENCY_WEIGHT, 1.0
        )

    def test_blend_stays_in_range(self) -> None:
        self.assertAlmostEqual(scoring.blended_score(1.0, 1.0), 1.0)
        self.assertAlmostEqual(scoring.blended_score(0.0, 0.0), 0.0)

    def test_relevance_outweighs_recency(self) -> None:
        """The 0.7/0.3 split has to actually mean something: a much better
        match must beat a much fresher one.
        """
        very_relevant_but_stale = scoring.blended_score(1.0, 0.0)
        barely_relevant_but_fresh = scoring.blended_score(0.0, 1.0)
        self.assertGreater(very_relevant_but_stale, barely_relevant_but_fresh)

    def test_importance_is_genuinely_absent_not_faked(self) -> None:
        """Guards the decision recorded in scoring.IMPORTANCE_NOTE.

        If someone adds a NULL-defaulted-to-1.0 importance term, blended_score
        would need a third argument and this call would break — which is the
        point. A silently multiplied constant would be undetectable.
        """
        import inspect

        params = inspect.signature(scoring.blended_score).parameters
        self.assertEqual(list(params), ["relevance", "recency"])
        self.assertIn("importance", scoring.IMPORTANCE_NOTE)


@requires_ollama
class RecallTestCase(MemoryServerTestCase):
    """Shared fixture: five facts across two targets, with controlled
    last_seen_at and confidence so ranking inputs are deterministic.

    Seeded through the real remember_fact tool (so the embeddings are real and
    consistent with how facts actually arrive), then last_seen_at/confidence
    are adjusted directly — those two are ranking inputs the tool does not let
    a caller set, and the tests need them pinned.
    """

    FIXTURE = [
        # (target_id, entity_key, content, last_seen_at, confidence)
        ("uni_a", "tuition", "Annual tuition fee is AUD 45,000 for international students",
         "2026-08-24T00:00:00.000Z", 0.9),
        ("uni_a", "intake", "Intake dates: February and July each year",
         "2026-08-24T00:00:00.000Z", 0.3),
        ("uni_a", "library", "The campus library is open 24 hours during exam period",
         "2026-08-24T00:00:00.000Z", None),
        ("job_b", "salary", "Base salary AUD 165,000 plus superannuation",
         "2026-08-24T00:00:00.000Z", 0.9),
        ("job_b", "skills", "Experience with Python, PyTorch and distributed training",
         "2026-08-24T00:00:00.000Z", 0.9),
    ]

    async def seed(self, client) -> None:
        for target_id, entity_key, content, last_seen, confidence in self.FIXTURE:
            result = await client.call_tool("remember_fact", {
                "target_id": target_id,
                "entity_key": entity_key,
                "content": content,
            })
            self.assertFalse(result.is_error, result.content)

        with db.transaction(self.db_path) as conn:
            for target_id, entity_key, _, last_seen, confidence in self.FIXTURE:
                conn.execute(
                    "UPDATE facts SET last_seen_at = ?, confidence = ? "
                    "WHERE target_id = ? AND entity_key = ? AND is_current = 1",
                    (last_seen, confidence, target_id, entity_key),
                )

    async def recall(self, client, **kwargs):
        payload = {"query": "how much are the fees"}
        payload.update(kwargs)
        return await client.call_tool("recall_facts", payload)

    @staticmethod
    def facts_of(result) -> list:
        return result.structured_content["result"]

    @staticmethod
    def keys_of(result) -> list:
        return [f["entity_key"] for f in result.structured_content["result"]]


class TestRecallRanking(RecallTestCase):
    async def test_semantically_close_facts_rank_above_the_rest(self) -> None:
        """The core retrieval claim. Query about money must surface the
        tuition and salary facts above library/intake/skills.

        The query wording here is MEASURED, not arbitrary. All five fixture
        facts share a last_seen_at, so this is pure relevance ordering, and
        nomic-embed-text turns out to be sensitive to phrasing in a way worth
        knowing about: of five natural phrasings tried, only this one ranks the
        money pair correctly. "what are the AUD amounts" ranks `intake` FIRST
        despite it containing no amounts at all. See notes/sprint-2.md Step 6
        — that limitation is recorded there rather than papered over here, and
        this test deliberately does not weaken its assertion to accommodate it.
        """
        async with Client(self.mcp) as client:
            await self.seed(client)
            result = await self.recall(
                client, query="how much money is involved", limit=5
            )

        self.assertFalse(result.is_error, result.content)
        keys = self.keys_of(result)
        self.assertEqual(len(keys), 5)

        money_facts = {"tuition", "salary"}
        top_two = set(keys[:2])
        self.assertEqual(
            top_two, money_facts,
            f"expected tuition+salary on top, got ranking {keys}",
        )

    async def test_limit_is_a_true_cap_on_final_results(self) -> None:
        async with Client(self.mcp) as client:
            await self.seed(client)
            result = await self.recall(client, limit=2)

        self.assertEqual(len(self.facts_of(result)), 2)

    async def test_results_are_sorted_by_score_descending(self) -> None:
        async with Client(self.mcp) as client:
            await self.seed(client)
            result = await self.recall(client, limit=5)

        scores = [f["score"] for f in self.facts_of(result)]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestRecallScoreComponents(RecallTestCase):
    """The components must mean something on their own, not just blend into a
    plausible-looking final number. Sprint 6/7 debugging depends on them.
    """

    async def test_relevance_is_higher_for_the_closer_match(self) -> None:
        async with Client(self.mcp) as client:
            await self.seed(client)
            result = await self.recall(
                client, query="annual tuition fee for international students",
                target_id="uni_a", limit=3,
            )

        facts = {f["entity_key"]: f for f in self.facts_of(result)}
        self.assertGreater(
            facts["tuition"]["relevance"], facts["library"]["relevance"],
            "the fact the query paraphrases must score higher on relevance",
        )

    async def test_recency_is_higher_for_the_more_recently_seen_fact(self) -> None:
        """Two facts, identical except for last_seen_at."""
        async with Client(self.mcp) as client:
            await self.seed(client)
            with db.transaction(self.db_path) as conn:
                conn.execute(
                    "UPDATE facts SET last_seen_at = ? WHERE entity_key = ?",
                    ("2026-01-01T00:00:00.000Z", "intake"),
                )
            result = await self.recall(client, query="dates and fees",
                                       target_id="uni_a", limit=5)

        facts = {f["entity_key"]: f for f in self.facts_of(result)}
        self.assertGreater(
            facts["tuition"]["recency"], facts["intake"]["recency"],
            "the fact seen in August must be fresher than the one from January",
        )

    async def test_components_are_in_range_and_compose_into_the_score(self) -> None:
        async with Client(self.mcp) as client:
            await self.seed(client)
            result = await self.recall(client, limit=5)

        for fact in self.facts_of(result):
            with self.subTest(entity_key=fact["entity_key"]):
                self.assertGreaterEqual(fact["relevance"], 0.0)
                self.assertLessEqual(fact["relevance"], 1.0)
                self.assertGreaterEqual(fact["recency"], 0.0)
                self.assertLessEqual(fact["recency"], 1.0)
                self.assertAlmostEqual(
                    fact["score"],
                    scoring.blended_score(fact["relevance"], fact["recency"]),
                    places=9,
                )


class TestRecallTargetFilter(RecallTestCase):
    async def test_target_filter_excludes_other_targets_entirely(self) -> None:
        """A query that matches strongly in uni_a must return NOTHING from
        uni_a when filtered to job_b — even though job_b holds
        semantically-adjacent money content (salary).
        """
        async with Client(self.mcp) as client:
            await self.seed(client)
            unfiltered = await self.recall(
                client, query="annual tuition fee international students", limit=5
            )
            filtered = await self.recall(
                client, query="annual tuition fee international students",
                target_id="job_b", limit=5,
            )

        self.assertIn("tuition", self.keys_of(unfiltered))

        targets = {f["target_id"] for f in self.facts_of(filtered)}
        self.assertEqual(targets, {"job_b"})
        self.assertNotIn("tuition", self.keys_of(filtered))

    async def test_narrowing_happens_pre_knn_not_as_a_post_hoc_trim(self) -> None:
        """THE distinction Part A's prefilter finding exists to protect.

        job_b holds exactly 2 facts. A post-hoc implementation would run KNN
        with k=limit over ALL facts, then drop the non-job_b rows — returning
        fewer than 2 whenever uni_a rows crowded the top. Asking for limit=2
        with a query that strongly favours uni_a must still return BOTH job_b
        facts.
        """
        async with Client(self.mcp) as client:
            await self.seed(client)
            result = await self.recall(
                client,
                query="annual tuition fee for international students at university",
                target_id="job_b",
                limit=2,
            )

        self.assertEqual(
            len(self.facts_of(result)), 2,
            "limit=2 over a 2-fact target returned fewer than 2 — the target "
            "filter is trimming an already-limited KNN result instead of "
            "narrowing the candidate pool before it",
        )
        self.assertEqual(set(self.keys_of(result)), {"salary", "skills"})


class TestRecallIsCurrentOnly(RecallTestCase):
    async def test_superseded_versions_are_never_returned(self) -> None:
        """Both versions have embeddings in facts_vec (Step 5 keeps history's
        vectors), so only the SQL filter prevents the old one surfacing.
        """
        async with Client(self.mcp) as client:
            await self.seed(client)
            modified = await client.call_tool("remember_fact", {
                "target_id": "uni_a",
                "entity_key": "tuition",
                "content": "Annual tuition fee increased to AUD 48,000 for 2027",
            })
            self.assertEqual(
                modified.structured_content["change_kind"], "modified"
            )
            result = await self.recall(
                client, query="tuition fee", target_id="uni_a", limit=5
            )

        # Sanity: the superseded row and BOTH vectors really do still exist.
        with db.connect(self.db_path) as conn:
            total = conn.execute(
                "SELECT COUNT(*) AS n FROM facts WHERE entity_key = 'tuition'"
            ).fetchone()["n"]
            vectors = conn.execute("SELECT COUNT(*) AS n FROM facts_vec").fetchone()["n"]
        self.assertEqual(total, 2, "history should be preserved")
        self.assertEqual(vectors, 6, "both versions should have embeddings")

        returned_ids = [f["fact_id"] for f in self.facts_of(result)]
        old_id = modified.structured_content["previous_fact_id"]
        new_id = modified.structured_content["fact_id"]

        self.assertIn(new_id, returned_ids)
        self.assertNotIn(old_id, returned_ids)

        contents = [f["content"] for f in self.facts_of(result)]
        self.assertTrue(any("48,000" in c for c in contents))
        self.assertFalse(any("45,000" in c for c in contents))


class TestRecallMinConfidence(RecallTestCase):
    async def test_null_confidence_passes_the_filter(self) -> None:
        """DECISION: NULL confidence PASSES any min_confidence threshold.

        Reason: NULL means "not assessed", not "assessed as zero". Letting
        SQL's default NULL-comparison semantics apply would silently drop every
        fact written before confidence scoring exists, which is all of them
        today.

        Fixture: tuition=0.9, intake=0.3, library=NULL. At min_confidence=0.5
        the 0.9 and the NULL return; the 0.3 does not.
        """
        async with Client(self.mcp) as client:
            await self.seed(client)
            result = await self.recall(
                client, query="university information", target_id="uni_a",
                limit=5, min_confidence=0.5,
            )

        keys = set(self.keys_of(result))
        self.assertEqual(
            keys, {"tuition", "library"},
            "expected the 0.9 and the NULL-confidence facts, not the 0.3",
        )

    async def test_zero_threshold_returns_everything(self) -> None:
        async with Client(self.mcp) as client:
            await self.seed(client)
            result = await self.recall(
                client, query="university information", target_id="uni_a",
                limit=5, min_confidence=0.0,
            )
        self.assertEqual(len(self.facts_of(result)), 3)

    async def test_threshold_above_every_score_still_keeps_nulls(self) -> None:
        """Makes the NULL decision unmistakable: at 0.95 nothing scored passes,
        but the unassessed fact still does.
        """
        async with Client(self.mcp) as client:
            await self.seed(client)
            result = await self.recall(
                client, query="university information", target_id="uni_a",
                limit=5, min_confidence=0.95,
            )
        self.assertEqual(set(self.keys_of(result)), {"library"})


class TestRecallEmptyAndInvalid(MemoryServerTestCase):
    async def test_empty_query_is_rejected(self) -> None:
        async with Client(self.mcp) as client:
            result = await client.call_tool("recall_facts", {"query": ""})

        self.assertTrue(result.is_error)
        self.assertIn("empty", result.content[0].text.lower())
        self.assertEqual(self.embed_call_count(), 0)

    async def test_whitespace_query_is_rejected(self) -> None:
        async with Client(self.mcp) as client:
            result = await client.call_tool("recall_facts", {"query": "  \n\t "})

        self.assertTrue(result.is_error)
        self.assertIn("empty", result.content[0].text.lower())
        self.assertEqual(self.embed_call_count(), 0)

    async def test_empty_facts_table_returns_empty_list_not_an_error(self) -> None:
        """NEGATIVE CONTROL. A first run against a new target legitimately
        knows nothing — that is an empty answer, not a failure. It must also
        not cost an embedding call, since there is nothing to compare against.
        """
        async with Client(self.mcp) as client:
            result = await client.call_tool(
                "recall_facts", {"query": "anything at all"}
            )

        self.assertFalse(result.is_error, result.content)
        self.assertEqual(result.structured_content["result"], [])
        self.assertEqual(
            self.embed_call_count(), 0,
            "no candidates means no reason to embed the query",
        )

    @requires_ollama
    async def test_target_with_zero_facts_returns_empty_list(self) -> None:
        async with Client(self.mcp) as client:
            await client.call_tool("remember_fact", {
                "target_id": "uni_a", "entity_key": "fees",
                "content": "Tuition is AUD 45,000",
            })
            result = await client.call_tool("recall_facts", {
                "query": "tuition", "target_id": "nonexistent_target",
            })

        self.assertFalse(result.is_error, result.content)
        self.assertEqual(result.structured_content["result"], [])

    async def test_limit_below_one_is_rejected(self) -> None:
        async with Client(self.mcp) as client:
            result = await client.call_tool(
                "recall_facts", {"query": "anything", "limit": 0}
            )
        self.assertTrue(result.is_error)


class TestStartupAssertion(unittest.TestCase):
    """Refusing to start is the correct behaviour, and it must be loud."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_unmigrated_database_exits_non_zero_with_a_clear_message(self) -> None:
        """A database file that exists but has no schema_migrations table."""
        import sqlite3

        unmigrated = self.dir / "empty.db"
        sqlite3.connect(unmigrated).close()

        with self.assertRaises(SystemExit) as caught:
            memory.assert_migrated(unmigrated)

        message = str(caught.exception)
        self.assertIn("schema_migrations", message)
        self.assertIn("migrate", message)

    def test_missing_database_exits_rather_than_creating_one(self) -> None:
        """The dangerous failure: silently creating an empty database.

        db.connect() calls mkdir and sqlite3.connect(), both of which happily
        create. If this server ever starts migrating or auto-creating, the
        orchestrator's sole ownership of the schema is gone.
        """
        missing = self.dir / "does_not_exist.db"

        with self.assertRaises(SystemExit) as caught:
            memory.assert_migrated(missing)

        self.assertIn("does not exist", str(caught.exception))
        self.assertFalse(missing.exists(), "the server created a database file")

    def test_partially_migrated_database_names_what_is_missing(self) -> None:
        """001-only database: the table exists, 002 does not."""
        from argus.storage.migrate import MIGRATIONS_DIR

        partial = self.dir / "partial.db"
        sql = (MIGRATIONS_DIR / "001_initial.sql").read_text(encoding="utf-8")
        with db.connect(partial) as conn:
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO schema_migrations (filename) VALUES (?)",
                ("001_initial.sql",),
            )

        with self.assertRaises(SystemExit) as caught:
            memory.assert_migrated(partial)

        message = str(caught.exception)
        self.assertIn(memory.REQUIRED_MIGRATION, message)
        self.assertIn("001_initial.sql", message)

    def test_fully_migrated_database_passes(self) -> None:
        """Control: proves the assertion above can actually be satisfied."""
        good = self.dir / "good.db"
        apply_migrations(good)
        memory.assert_migrated(good)  # must not raise


class TestStdoutDiscipline(MemoryServerTestCase):
    """A stdio MCP server owns stdout. One stray byte corrupts the frame stream.

    The failure this prevents is nasty precisely because it is silent: Inspector
    and the orchestrator both just stop working, with no error pointing at the
    print() that caused it.
    """

    async def test_a_full_tool_call_writes_nothing_to_stdout(self) -> None:
        captured = io.StringIO()
        with redirect_stdout(captured):
            async with Client(self.mcp) as client:
                await self.remember(client, content="")  # rejected path
                if OLLAMA_UP:
                    await self.remember(client)  # success path

        self.assertEqual(captured.getvalue(), "")

    def test_log_writes_to_stderr_not_stdout(self) -> None:
        captured = io.StringIO()
        with redirect_stdout(captured):
            memory.log("this must not reach stdout")
        self.assertEqual(captured.getvalue(), "")


@requires_ollama
class TestEveryToolReturnsStructuredOutput(MemoryServerTestCase):
    """Regression guard against the exact bug found on recall_facts: under
    `from __future__ import annotations`, writing an already-redundant quoted
    return type (`-> "list[RecalledFact]"`) becomes a DOUBLY-quoted string
    pydantic cannot resolve. The tool still registers and still answers, but
    `structured_content` comes back None — silent to everything except a
    caller that actually inspects it, which `is_error` checks alone do not.

    Generic on purpose, not one test per tool: it discovers the registered
    tool list from the server itself and iterates it, so tool 3, 4, and beyond
    are covered automatically. VALID_PAYLOADS deliberately has no default —
    a tool with no entry fails LOUDLY (KeyError) rather than being silently
    skipped, so this cannot go quietly stale as tools are added.
    """

    def valid_payloads(self) -> dict:
        return {
            "remember_fact": {
                "target_id": TARGET,
                "entity_key": "structured_output_probe",
                "content": "Probe content for the structured-output regression test.",
            },
            "recall_facts": {"query": "probe query"},
        }

    async def test_every_tool_returns_non_null_structured_content(self) -> None:
        payloads = self.valid_payloads()

        async with Client(self.mcp) as client:
            tools = await client.list_tools()
            tool_names = [t.name for t in tools.tools]
            self.assertTrue(tool_names, "no tools registered - nothing to test")

            for name in tool_names:
                with self.subTest(tool=name):
                    self.assertIn(
                        name, payloads,
                        f"tool {name!r} has no entry in valid_payloads() - add one, "
                        f"do not let a new tool go untested",
                    )
                    result = await client.call_tool(name, payloads[name])
                    self.assertFalse(
                        result.is_error, f"{name} failed: {result.content}"
                    )
                    self.assertIsNotNone(
                        result.structured_content,
                        f"{name} returned structured_content=None - this is the "
                        f"doubly-quoted-annotation bug (or equivalent): the tool "
                        f"answers but the model-facing structured result is empty",
                    )


class TestContentHash(unittest.TestCase):
    """Normalization is a contract: changing it reclassifies every stored fact
    as 'modified' on its next run.
    """

    def test_surrounding_whitespace_is_ignored(self) -> None:
        self.assertEqual(
            memory.content_hash("  hello  ", None), memory.content_hash("hello", None)
        )

    def test_none_and_empty_attributes_hash_the_same(self) -> None:
        self.assertEqual(
            memory.content_hash("hello", None), memory.content_hash("hello", {})
        )

    def test_attribute_key_order_does_not_matter(self) -> None:
        self.assertEqual(
            memory.content_hash("hello", {"a": "1", "b": "2"}),
            memory.content_hash("hello", {"b": "2", "a": "1"}),
        )

    def test_interior_whitespace_and_case_are_real_changes(self) -> None:
        self.assertNotEqual(
            memory.content_hash("Fees: $45,000", None),
            memory.content_hash("fees: $45,000", None),
        )

    def test_attribute_values_affect_the_hash(self) -> None:
        self.assertNotEqual(
            memory.content_hash("hello", {"a": "1"}),
            memory.content_hash("hello", {"a": "2"}),
        )

    def test_content_and_attribute_split_cannot_collide(self) -> None:
        """The \\x00 separator earns its place here."""
        self.assertNotEqual(
            memory.content_hash('x{"a":"1"}', None),
            memory.content_hash("x", {"a": "1"}),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
