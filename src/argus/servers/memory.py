"""Argus memory MCP server.

Run: python -m argus.servers.memory --db-path <path> --run-id <id>
     (with src/ on PYTHONPATH)
     python src/argus/servers/memory.py --db-path <path> --run-id <id>

The `-m` form needs `src/` on PYTHONPATH because this repo has no editable
install (`pip install -e .`) — `-m` resolves the module against sys.path
before any of this file's own code runs, so the sys.path bootstrap two lines
down cannot rescue it the way it rescues direct-script execution. Confirmed:
`python -m argus.servers.memory ...` run bare from the repo root fails with
`ModuleNotFoundError: No module named 'argus'`; the identical failure
reproduces on `runner.py` and `migrate.py`'s own documented `-m` commands
run bare — this is a repo-wide gap, not specific to this module. Matches the
convention `migrate.py` already documents and `crash_test.py` already relies
on when it spawns `runner.py` (`env["PYTHONPATH"] = str(SRC) + ...`).

Spawned per run by the orchestrator over stdio. `--run-id` is how embedding
calls become attributable in `llm_calls` without putting a run id into the
model-facing tool schema (see design-decisions.md).

STDOUT BELONGS TO THE MCP PROTOCOL. Never `print()` here. Every diagnostic
goes to stderr — a single stray byte on stdout corrupts the JSON-RPC frame
stream and both MCP Inspector and the orchestrator fail in ways that look like
anything except "something wrote to stdout".

This server NEVER runs migrations. It asserts migration 002 is already applied
and exits non-zero if not. Only the orchestrator migrates (CLAUDE.md, Storage
concurrency).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Optional

if __package__ in (None, ""):  # direct-script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp.server import MCPServer  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from argus import embeddings  # noqa: E402
from argus.servers import scoring  # noqa: E402
from argus.storage import db  # noqa: E402

SERVER_NAME = "argus-memory"

#: The server refuses to start unless this migration is recorded as applied.
REQUIRED_MIGRATION = "002_facts_approvals.sql"


# ---------------------------------------------------------------------------
# logging — stderr only, see module docstring
# ---------------------------------------------------------------------------


def log(message: str) -> None:
    """The only output channel this module may use besides MCP frames."""
    print(f"[{SERVER_NAME}] {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# result schema
# ---------------------------------------------------------------------------


class RememberResult(BaseModel):
    """What the caller gets back. Deliberately small and literal.

    `change_kind` here is the API's vocabulary, which is NOT identical to the
    `facts.change_kind` column's CHECK enum ('added'|'modified'|'removed').
    'unchanged' is a real outcome for a caller but never a stored row — an
    unchanged fact touches last_seen_* on the existing row and writes nothing
    new. Do not "fix" this by widening the column's CHECK.
    """

    fact_id: int = Field(description="Row id of the current fact after this call.")
    change_kind: Literal["added", "modified", "unchanged"]
    version: int
    previous_fact_id: Optional[int] = Field(
        default=None,
        description="The row this one superseded. Null unless change_kind is 'modified'.",
    )


class RecalledFact(BaseModel):
    """One retrieved fact plus the score components that ranked it.

    `relevance` and `recency` are exposed alongside the blended `score` on
    purpose: they are the only way to answer "why did this rank here?" when
    Sprint 6/7 debugging turns up a bad ordering. A single opaque number would
    make that undiagnosable.
    """

    fact_id: int
    target_id: str
    entity_key: str
    content: str
    attributes: Optional[dict] = None
    score: float = Field(description="Blended rank score, higher is better.")
    relevance: float = Field(description="Semantic closeness to the query, 0-1.")
    recency: float = Field(description="Freshness of last_seen_at, 0-1.")
    last_seen_at: str
    source_url: Optional[str] = None


class EmptyContentError(ValueError):
    """Raised for blank content. Surfaces to the model as a readable message.

    The gate deliberately left out of embeddings.py in Step 4: an empty string
    embeds perfectly happily into a unit vector that is indistinguishable from
    a real fact downstream (measured — see notes/sprint-2.md). Something has to
    say no, and it has to be the caller-facing layer, because only here is it
    knowable that a blank fact is a mistake rather than a legitimate vector.
    """


class EmptyQueryError(ValueError):
    """Raised for a blank recall query. Same reasoning as EmptyContentError.

    A whitespace query embeds into a perfectly valid vector that will happily
    return the five facts nearest to nothing in particular. Silently answering
    a question nobody asked is worse than refusing.
    """


class InvalidTimestampError(ValueError):
    """Raised when `since` is not an unambiguous, UTC, ISO 8601 timestamp.

    Deliberately strict: a bad `since` is rejected, never silently coerced to
    "now" or to epoch. Either of those would make list_changes_since return a
    plausible-looking but wrong answer (everything, or nothing) instead of an
    error the caller can actually notice and fix.
    """


class FactChange(BaseModel):
    """One row from the change feed. Mirrors a single facts row, not a diff
    object synthesized from two rows — the row already IS the change.
    """

    fact_id: int
    target_id: str
    entity_key: str
    change_kind: Literal["added", "modified", "removed"]
    version: int
    content: str
    previous_content: Optional[str] = Field(
        default=None,
        description=(
            "Content of the row this one superseded, via supersedes_id. "
            "Null for 'added' rows, which have no prior version."
        ),
    )
    observed_at: str = Field(
        description="created_at of this fact version - when THIS row came "
        "into being, not when it was last seen."
    )


# ---------------------------------------------------------------------------
# startup
# ---------------------------------------------------------------------------


def assert_migrated(db_path: Path) -> None:
    """Refuse to start unless migration 002 is applied. Never migrate here.

    Reads the `schema_migrations` table, which is the real migration marker.
    `PRAGMA user_version` is NOT used by this project and reads 0 even on a
    fully migrated database, so anything trusting it would conclude nothing had
    run (see notes/sprint-2.md, Step 3).
    """
    if not Path(db_path).exists():
        raise SystemExit(
            f"{SERVER_NAME}: database does not exist: {db_path}\n"
            f"The orchestrator creates and migrates it. This server never does."
        )

    try:
        with db.connect(db_path) as conn:
            rows = conn.execute("SELECT filename FROM schema_migrations").fetchall()
    except sqlite3.OperationalError as exc:
        raise SystemExit(
            f"{SERVER_NAME}: cannot read schema_migrations from {db_path}: {exc}\n"
            f"Run the orchestrator's migrations first: "
            f"python -m argus.storage.migrate"
        ) from exc

    applied = {row["filename"] for row in rows}
    if REQUIRED_MIGRATION not in applied:
        raise SystemExit(
            f"{SERVER_NAME}: required migration {REQUIRED_MIGRATION} is not applied "
            f"to {db_path}.\n"
            f"Applied: {sorted(applied) or '(none)'}\n"
            f"This server does not run migrations - that is the orchestrator's job. "
            f"Run: python -m argus.storage.migrate"
        )


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------


def content_hash(content: str, attributes: Optional[dict]) -> str:
    """sha256 over NORMALIZED content plus attributes.

    "Normalized" is defined here, once, and must never drift — a change to this
    function silently reclassifies every existing fact as 'modified' on its
    next run, producing a spurious new version of everything.

      - content: .strip() only. Interior whitespace and case are preserved,
        because "Fees: $45,000" and "fees: $45,000" are a real change worth
        seeing on a monitored page.
      - attributes: json.dumps(sort_keys=True), so key order from the extractor
        cannot fake a change. None and {} hash identically - an extractor that
        returns no attributes and one that returns an empty dict mean the same
        thing.
      - the two are joined with a \\x00 separator so that content ending in
        what looks like JSON cannot collide with a different content/attributes
        split.
    """
    normalized_content = content.strip()
    normalized_attributes = json.dumps(attributes or {}, sort_keys=True, separators=(",", ":"))
    payload = f"{normalized_content}\x00{normalized_attributes}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _format_iso(dt: datetime) -> str:
    """Render a datetime in the schema's own DEFAULT format.

    Millisecond precision, trailing 'Z'. Every timestamp this module writes or
    compares against goes through this one function, so `created_at` values
    written by SQLite's own DEFAULT and timestamps written or compared by
    Python code are always the same shape - which is what makes a plain
    string comparison in SQL (`created_at > ?`) a valid substitute for parsing
    both sides as datetimes.
    """
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def utcnow_iso() -> str:
    """Matches the format the schema's own DEFAULT produces."""
    return _format_iso(datetime.now(timezone.utc))


def parse_since(value: str) -> str:
    """Validate `since` as an unambiguous UTC ISO 8601 timestamp and return it
    RE-FORMATTED to the schema's canonical shape.

    Raises InvalidTimestampError rather than coercing, on three distinct
    failures - each reported with what was actually wrong, not just "invalid":
      - not parseable as ISO 8601 at all
      - parseable but missing timezone information (ambiguous: whose local
        time is it?)
      - parseable with an explicit non-UTC offset (this project has exactly
        one clock, UTC, everywhere else - accepting silent offset conversion
        here would be the one place that assumption quietly stopped holding)

    Re-formatting rather than comparing the raw input string against
    `created_at` in SQL matters: a caller may reasonably pass
    "2026-08-24T00:00:00+00:00" instead of "...000Z", and a raw string
    comparison would silently misorder against the schema's own
    "...000Z"-suffixed values despite meaning the exact same instant.
    """
    if not isinstance(value, str) or not value.strip():
        raise InvalidTimestampError(
            f"'since' must be a non-empty ISO 8601 timestamp string, got {value!r}"
        )

    candidate = value.strip()
    try:
        parsed = datetime.fromisoformat(
            candidate[:-1] + "+00:00" if candidate.endswith("Z") else candidate
        )
    except ValueError as exc:
        raise InvalidTimestampError(
            f"'since' is not a valid ISO 8601 timestamp: {value!r} ({exc})"
        ) from exc

    if parsed.tzinfo is None:
        raise InvalidTimestampError(
            f"'since' must include a UTC timezone marker ('Z' or '+00:00'), "
            f"got a timezone-naive value: {value!r}"
        )
    if parsed.utcoffset() != timedelta(0):
        raise InvalidTimestampError(
            f"'since' must be UTC (offset +00:00), got offset "
            f"{parsed.utcoffset()} in {value!r}"
        )

    return _format_iso(parsed.astimezone(timezone.utc))


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------


class MemoryServer:
    """Holds the two things every tool needs: where the database is, and which
    run to attribute embedding calls to.

    A class rather than module globals so the tests can build one per temp
    database without leaking state between them.
    """

    def __init__(self, db_path: Path, run_id: str):
        self.db_path = Path(db_path)
        self.run_id = run_id

    async def remember_fact(
        self,
        target_id: str,
        entity_key: str,
        content: str,
        attributes: Optional[dict] = None,
        source_url: Optional[str] = None,
        confidence: float = 1.0,
    ) -> RememberResult:
        """Record a fact, resolving to added / modified / unchanged.

        Reads the current row TWICE: once before opening a transaction
        (advisory only, used solely to decide whether embedding is worth doing
        up front) and once again inside `db.atomic()`, which is the sole
        authority for the actual write decision. Do not collapse these into
        one lookup — that is exactly the change that would reintroduce the
        race two concurrent callers for the same entity_key are protected
        against today (see TestConcurrentRememberFact in
        tests/test_memory_server.py for the proof this holds under real
        overlap).
        """
        if not content or not content.strip():
            raise EmptyContentError(
                "content must not be empty or whitespace-only. "
                "An empty string embeds into a valid-looking vector that cannot "
                "be told apart from a real fact later, so it is rejected here "
                "rather than stored. Pass the actual extracted text, or do not "
                "call remember_fact at all if there is nothing to record."
            )

        digest = content_hash(content, attributes)
        now = utcnow_iso()

        # One connection per call, NOT one shared by the server. Two concurrent
        # remember_fact invocations must be able to hold two separate
        # transactions and contend on the write lock; a shared connection would
        # instead fail with "cannot start a transaction within a transaction",
        # because the `await` on the embed below yields the event loop and lets
        # a second invocation reach BEGIN IMMEDIATE while the first is open.
        with db.connect(self.db_path) as conn:
            # A cheap read OUTSIDE the transaction, used only to decide whether
            # an embedding is needed. It is deliberately NOT trusted for the
            # write decision - it is re-read inside the transaction below.
            # Reading here is safe because the worst case is embedding a vector
            # we end up not needing, which costs ~0.9s and nothing else.
            existing = self._current_row(conn, target_id, entity_key)

            vector_blob = None
            if existing is None or existing["content_hash"] != digest:
                # Embed BEFORE opening the transaction. CLAUDE.md: never hold a
                # transaction open across an Ollama call - the write lock is
                # database-wide and an embed takes ~0.9s, which would block the
                # orchestrator process for that whole time.
                vector = await embeddings.embed(
                    content.strip(), run_id=self.run_id, conn=conn
                )
                vector_blob = embeddings.serialize(vector)

            # THE transaction. The current-row lookup is repeated inside it and
            # that repetition is the point: the read and the write must be one
            # atomic unit. If the lookup outside were trusted, two concurrent
            # calls for the same (target_id, entity_key) could both read "no
            # current row", both decide 'added', and both INSERT with
            # is_current=1 - which the partial unique index would reject as an
            # IntegrityError, turning a benign race into a caller-visible
            # failure. Re-reading under BEGIN IMMEDIATE means the second caller
            # sees the first caller's committed row and correctly resolves to
            # 'unchanged' or 'modified' instead.
            with db.atomic(conn):
                current = self._current_row(conn, target_id, entity_key)

                if current is None:
                    return self._insert_new(
                        conn,
                        target_id=target_id,
                        entity_key=entity_key,
                        content=content,
                        attributes=attributes,
                        digest=digest,
                        source_url=source_url,
                        confidence=confidence,
                        now=now,
                        vector_blob=vector_blob,
                        version=1,
                        supersedes_id=None,
                        change_kind="added",
                    )

                if current["content_hash"] == digest:
                    # Unchanged. Touch last_seen_* on the EXISTING row and stop.
                    # No new row, no re-embed - the vector cannot have changed
                    # if the content hash did not.
                    conn.execute(
                        "UPDATE facts SET last_seen_run_id = ?, last_seen_at = ? "
                        "WHERE fact_id = ?",
                        (self.run_id, now, current["fact_id"]),
                    )
                    return RememberResult(
                        fact_id=current["fact_id"],
                        change_kind="unchanged",
                        version=current["version"],
                        previous_fact_id=None,
                    )

                # Modified: retire the old row, then insert the new version.
                # Order matters - the partial unique index permits exactly one
                # is_current=1 row per (target_id, entity_key), so the old row
                # must be cleared before the new one lands.
                conn.execute(
                    "UPDATE facts SET is_current = 0 WHERE fact_id = ?",
                    (current["fact_id"],),
                )
                return self._insert_new(
                    conn,
                    target_id=target_id,
                    entity_key=entity_key,
                    content=content,
                    attributes=attributes,
                    digest=digest,
                    source_url=source_url,
                    confidence=confidence,
                    now=now,
                    vector_blob=vector_blob,
                    version=current["version"] + 1,
                    supersedes_id=current["fact_id"],
                    change_kind="modified",
                    first_seen_run_id=current["first_seen_run_id"],
                    first_seen_at=current["first_seen_at"],
                )

    async def recall_facts(
        self,
        query: str,
        target_id: Optional[str] = None,
        limit: int = 5,
        min_confidence: float = 0.0,
    ) -> list[RecalledFact]:
        """Semantic search over CURRENT facts, ranked by relevance and recency.

        The candidate pool is narrowed BEFORE the KNN, not after it. Eligible
        fact_ids are selected from `facts` first (is_current, target_id,
        confidence) and passed into the vector query as a `fact_id IN (...)`
        prefilter — verified to work alongside MATCH at sqlite-vec 0.1.9. The
        post-hoc alternative (KNN top-k, then drop the wrong-target rows) makes
        `limit` a lie: asking for 5 could return 1 because the other 4 belonged
        to a different target.

        Every eligible candidate is scored, not just the KNN's top `limit`.
        That is required for correctness, not thoroughness: the final ranking
        blends recency, so a fact ranked 8th by pure distance can legitimately
        finish 1st. Truncating at the KNN would silently drop it before recency
        was ever considered. `limit` is applied last, to the fully sorted list.

        Query text is embedded with NO prefix, exactly as stored fact text is —
        see notes/sprint-2.md Step 6 Part A, where ten fact-shaped pairs picked
        unprefixed over both `search_document:` forms on mean margin and
        inversion count. There is deliberately no query/document asymmetry.
        """
        if not query or not query.strip():
            raise EmptyQueryError(
                "query must not be empty or whitespace-only. A blank query "
                "embeds into a valid vector and would return whichever facts "
                "happen to sit nearest to nothing in particular, which reads "
                "like an answer but is not one. Pass the actual question."
            )
        if limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")

        with db.connect(self.db_path) as conn:
            # Eligibility, resolved in SQL. is_current = 1 is enforced HERE, in
            # the query that builds the candidate pool - a superseded fact is
            # never a candidate, rather than being fetched and filtered out
            # afterwards.
            #
            # NULL confidence PASSES any min_confidence filter. Rationale:
            # confidence is nullable and unset means "not assessed", not
            # "assessed as zero". SQL's own `NULL >= 0.5` is NULL (falsy), so
            # relying on default behaviour would silently drop every fact whose
            # confidence was never scored - including everything written before
            # confidence scoring exists. COALESCE makes the choice explicit.
            sql = (
                "SELECT fact_id FROM facts "
                "WHERE is_current = 1 AND COALESCE(confidence, 1.0) >= ?"
            )
            params: list = [min_confidence]
            if target_id is not None:
                sql += " AND target_id = ?"
                params.append(target_id)

            eligible = [row["fact_id"] for row in conn.execute(sql, params)]
            if not eligible:
                # No candidates is a normal answer, not an error: a first run
                # against a new target legitimately knows nothing yet.
                return []

            # Embed OUTSIDE any transaction (CLAUDE.md) - though this path takes
            # no write lock at all, being read-only.
            query_vector = await embeddings.embed(
                query.strip(), run_id=self.run_id, conn=conn
            )
            query_blob = embeddings.serialize(query_vector)

            placeholders = ", ".join("?" for _ in eligible)
            rows = conn.execute(
                f"SELECT v.fact_id, v.distance, f.target_id, f.entity_key, "
                f"f.content, f.attributes, f.last_seen_at, f.source_url "
                f"FROM facts_vec v JOIN facts f ON f.fact_id = v.fact_id "
                f"WHERE v.embedding MATCH ? AND k = ? "
                f"AND v.fact_id IN ({placeholders})",
                [query_blob, len(eligible), *eligible],
            ).fetchall()

        scored = []
        for row in rows:
            score, relevance, recency = scoring.score_candidate(
                row["distance"], row["last_seen_at"]
            )
            scored.append(
                RecalledFact(
                    fact_id=row["fact_id"],
                    target_id=row["target_id"],
                    entity_key=row["entity_key"],
                    content=row["content"],
                    attributes=json.loads(row["attributes"]) if row["attributes"] else None,
                    score=score,
                    relevance=relevance,
                    recency=recency,
                    last_seen_at=row["last_seen_at"],
                    source_url=row["source_url"],
                )
            )

        # Sort AFTER scoring every candidate, then cap. Ties broken by fact_id
        # so the ordering is deterministic rather than dependent on row order.
        scored.sort(key=lambda f: (-f.score, f.fact_id))
        return scored[:limit]

    async def list_changes_since(
        self,
        since: str,
        target_id: Optional[str] = None,
        kinds: Optional[list] = None,
        limit: int = 50,
    ) -> list[FactChange]:
        """The change feed: every fact VERSION whose own created_at falls
        after `since`, most recent first.

        Ground truth for debugging recall quality: this tool never calls
        embed(), never touches facts_vec, and writes zero llm_calls rows - it
        is the one memory-server tool fully unaffected by the retrieval
        quality risk documented in Step 6 (notes/sprint-2.md). If a fact's
        history looks wrong, the bug is in remember_fact or upstream of it,
        never in embedding quality, because this tool never embeds anything.

        THE created_at DECISION. "Changed since X" is answered against each
        row's OWN created_at - the moment that specific version came into
        being - not last_seen_at. last_seen_at moves every time an unchanged
        fact is touched again (Step 5's 'unchanged' path bumps it without
        writing a new row), so a fact from months ago that was simply
        re-observed yesterday would look like a change today if last_seen_at
        were used. created_at is written once, at INSERT, and never updated -
        it is genuinely immobile, which is the property a change feed needs.

        Not scoped to is_current: a change feed shows history, including
        versions that have since been superseded again - unlike recall_facts,
        which deliberately shows only the current snapshot.

        previous_content: content of the row this one's supersedes_id points
        at, via a LEFT JOIN - not a change_kind-specific branch. This covers
        'added' (supersedes_id NULL -> previous_content NULL) and 'modified'
        uniformly, and extends the same rule to 'removed': whatever content
        existed right before removal is exactly the useful signal a removal
        entry should carry, and requires nothing removal-specific here. The
        one thing this means for Sprint 4: a 'removed' row must set
        supersedes_id, the same contract 'modified' already has, or its
        previous_content will silently read NULL.
        """
        canonical_since = parse_since(since)

        sql = (
            "SELECT f.fact_id, f.target_id, f.entity_key, f.change_kind, "
            "f.version, f.content, f.created_at, prev.content AS previous_content "
            "FROM facts f LEFT JOIN facts prev ON prev.fact_id = f.supersedes_id "
            "WHERE f.created_at > ?"
        )
        params: list = [canonical_since]

        # In the WHERE clause directly: no KNN step here to worry about
        # ordering the filter around, unlike recall_facts.
        if target_id is not None:
            sql += " AND f.target_id = ?"
            params.append(target_id)

        if kinds is not None:
            if not kinds:
                return []
            placeholders = ", ".join("?" for _ in kinds)
            sql += f" AND f.change_kind IN ({placeholders})"
            params.extend(kinds)

        sql += " ORDER BY f.created_at DESC LIMIT ?"
        params.append(limit)

        with db.connect(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            FactChange(
                fact_id=row["fact_id"],
                target_id=row["target_id"],
                entity_key=row["entity_key"],
                change_kind=row["change_kind"],
                version=row["version"],
                content=row["content"],
                previous_content=row["previous_content"],
                observed_at=row["created_at"],
            )
            for row in rows
        ]

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _current_row(conn: sqlite3.Connection, target_id: str, entity_key: str) -> Any:
        return conn.execute(
            "SELECT fact_id, content_hash, version, first_seen_run_id, first_seen_at "
            "FROM facts WHERE target_id = ? AND entity_key = ? AND is_current = 1",
            (target_id, entity_key),
        ).fetchone()

    def _insert_new(
        self,
        conn: sqlite3.Connection,
        *,
        target_id: str,
        entity_key: str,
        content: str,
        attributes: Optional[dict],
        digest: str,
        source_url: Optional[str],
        confidence: float,
        now: str,
        vector_blob: Optional[bytes],
        version: int,
        supersedes_id: Optional[int],
        change_kind: str,
        first_seen_run_id: Optional[str] = None,
        first_seen_at: Optional[str] = None,
    ) -> RememberResult:
        """Insert the fact row and its vector. Caller must hold the transaction.

        first_seen_* carry forward from the superseded row on a modification:
        the ENTITY was first seen when version 1 landed, even though this
        particular row is new. Defaulting them to `now` would make every
        modification look like a brand-new discovery.
        """
        cursor = conn.execute(
            "INSERT INTO facts ("
            "target_id, entity_key, content, attributes, content_hash, version, "
            "is_current, supersedes_id, change_kind, confidence, source_url, "
            "first_seen_run_id, last_seen_run_id, first_seen_at, last_seen_at"
            ") VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                target_id,
                entity_key,
                content.strip(),
                json.dumps(attributes, sort_keys=True) if attributes else None,
                digest,
                version,
                supersedes_id,
                change_kind,
                confidence,
                source_url,
                first_seen_run_id or self.run_id,
                self.run_id,
                first_seen_at or now,
                now,
            ),
        )
        fact_id = int(cursor.lastrowid)

        if vector_blob is not None:
            # facts_vec is keyed by fact_id, so a new version gets its own row
            # and the superseded version's vector stays put. History keeps its
            # embeddings; nothing is overwritten.
            conn.execute(
                "INSERT INTO facts_vec (fact_id, embedding) VALUES (?, ?)",
                (fact_id, vector_blob),
            )

        return RememberResult(
            fact_id=fact_id,
            change_kind=change_kind if change_kind != "added" else "added",
            version=version,
            previous_fact_id=supersedes_id,
        )


def build_server(db_path: Path, run_id: str) -> MCPServer:
    """Wire a MemoryServer's methods onto an MCPServer instance.

    Built per-instance rather than with module-level decorators so tests can
    stand up an isolated server per temp database. The tool is registered with
    an explicit name so the model-facing surface never depends on Python
    method naming.
    """
    backend = MemoryServer(db_path, run_id)
    mcp = MCPServer(SERVER_NAME)

    @mcp.tool(
        name="remember_fact",
        description=(
            "Record a fact about a target. Returns whether it was added, "
            "modified, or unchanged since the last run. Facts are versioned: "
            "modifying one preserves the previous version rather than "
            "overwriting it."
        ),
    )
    async def remember_fact(
        target_id: str,
        entity_key: str,
        content: str,
        attributes: Optional[dict] = None,
        source_url: Optional[str] = None,
        confidence: float = 1.0,
    ) -> RememberResult:
        return await backend.remember_fact(
            target_id=target_id,
            entity_key=entity_key,
            content=content,
            attributes=attributes,
            source_url=source_url,
            confidence=confidence,
        )

    @mcp.tool(
        name="recall_facts",
        description=(
            "Search remembered facts by meaning, not keywords. Returns the "
            "best matches ranked by a blend of semantic relevance and how "
            "recently each fact was seen. Only current facts are searched; "
            "superseded versions are never returned."
        ),
    )
    async def recall_facts(
        query: str,
        target_id: Optional[str] = None,
        limit: int = 5,
        min_confidence: float = 0.0,
    ) -> list[RecalledFact]:
        return await backend.recall_facts(
            query=query,
            target_id=target_id,
            limit=limit,
            min_confidence=min_confidence,
        )

    @mcp.tool(
        name="list_changes_since",
        description=(
            "List facts that changed after a given UTC timestamp (ISO 8601), "
            "most recent first. Ground truth for what actually changed - "
            "does not use embeddings or semantic search."
        ),
    )
    async def list_changes_since(
        since: str,
        target_id: Optional[str] = None,
        kinds: Optional[list] = None,
        limit: int = 50,
    ) -> list[FactChange]:
        return await backend.list_changes_since(
            since=since, target_id=target_id, kinds=kinds, limit=limit
        )

    return mcp


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="argus.servers.memory")
    parser.add_argument("--db-path", required=True, help="path to the Argus SQLite file")
    parser.add_argument(
        "--run-id",
        required=True,
        help="run this server's embedding calls are attributed to in llm_calls",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db_path)
    assert_migrated(db_path)

    log(f"serving {db_path} for run {args.run_id}")
    build_server(db_path, args.run_id).run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
