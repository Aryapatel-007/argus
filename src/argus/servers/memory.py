"""Argus memory MCP server.

    python -m argus.servers.memory --db-path <path> --run-id <id>

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
from datetime import datetime, timezone
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


def utcnow_iso() -> str:
    """Matches the format the schema's own DEFAULT produces."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + (
        f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"
    )


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
