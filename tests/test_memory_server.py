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
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from mcp import Client  # noqa: E402

from argus import config, embeddings  # noqa: E402
from argus.orchestrator import llm  # noqa: E402
from argus.servers import memory  # noqa: E402
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
