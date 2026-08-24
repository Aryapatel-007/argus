"""Retrieval scoring for recall_facts.

    score = (RELEVANCE_WEIGHT * relevance) + (RECENCY_WEIGHT * recency)

Pure functions, no SQL and no I/O, so the ranking behaviour can be tested
without a database or an Ollama server. The memory server calls into this;
nothing here calls back out.

`importance` is deliberately ABSENT from the formula — see IMPORTANCE_NOTE.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# weights — named so Sprint 5 can retune without touching logic
# ---------------------------------------------------------------------------

#: Settled at the top of Sprint 2. Relevance dominates: a very old fact that
#: actually answers the query is more use than a fresh one that does not.
RELEVANCE_WEIGHT = 0.7
RECENCY_WEIGHT = 0.3

#: Days for recency to fall to 0.5. A PLACEHOLDER TUNABLE, not a measured
#: constant — unlike the relevance conversion below, nothing about this number
#: has been measured. 30 days is chosen to be roughly meaningful for targets
#: checked weekly: a fact seen last week still scores ~0.85, one from two
#: months ago ~0.25. Retune in Sprint 5 against real recall quality.
RECENCY_HALF_LIFE_DAYS = 30.0

IMPORTANCE_NOTE = """
`importance` exists as a column in migration 002 and stays NULL through
Sprint 2. It is NOT in this formula, and is not defaulted to 1.0 either.

Defaulting a NULL importance to 1.0 and multiplying by it produces exactly
the same ranking as leaving it out — every score would be multiplied by the
same constant, and the ordering, which is all that matters here, would be
identical. It would look like the feature existed while doing nothing, which
is worse than its honest absence: a future reader would assume importance was
already wired up and tested.

Sprint 5 plugs it in as a third weighted term alongside relevance and recency
(weights renormalised to sum to 1), at which point it gets its own tests.
"""


# ---------------------------------------------------------------------------
# relevance
# ---------------------------------------------------------------------------

#: The largest L2 distance possible between two unit vectors (antipodal).
MAX_UNIT_L2 = 2.0


def relevance_from_l2(distance: float) -> float:
    """Convert vec0's L2 distance into a bounded relevance score in [0, 1].

        relevance = 1 - (L2^2 / 4)

    Direction is correct by construction: distance 0 -> relevance 1.0, and the
    maximum possible distance (2.0, antipodal unit vectors) -> relevance 0.0.
    Closer always scores higher.

    WHY THIS CONVERSION IS VALID HERE SPECIFICALLY. `facts_vec` is declared
    without `distance_metric=`, so vec0 returns L2 (confirmed against the
    0.1.9 docs). L2 is unbounded in general, which would make any normalisation
    arbitrary — but nomic-embed-text returns UNIT-NORMALISED vectors (measured:
    norms 0.9999999288 to 1.0000000163 across real embeddings). For unit
    vectors:

        ||a - b||^2 = 2 - 2*cos(a, b)

    so L2 is bounded to [0, 2], and rearranging gives cos = 1 - L2^2/2. Mapping
    cosine's [-1, 1] onto [0, 1] with (cos + 1) / 2 yields exactly
    1 - L2^2/4. Verified against real vectors to ~1e-7 (see notes/sprint-2.md).

    This is an exact algebraic identity, not a heuristic squash — no tuning
    constant, nothing to retune later. It DOES depend on the embedding model
    producing unit vectors: swap `models.embed` for one that does not and this
    silently starts returning negative relevance for distant pairs, which is
    why the clamp below is not decoration.
    """
    if distance < 0:
        raise ValueError(f"L2 distance cannot be negative, got {distance!r}")

    relevance = 1.0 - (distance * distance) / 4.0

    # Only reachable if the vectors are not unit-normalised (a model swap) or
    # via float error at the extremes. Clamping keeps the score in contract
    # rather than letting a negative relevance quietly poison the blend.
    return max(0.0, min(1.0, relevance))


# ---------------------------------------------------------------------------
# recency
# ---------------------------------------------------------------------------


def _parse_iso(value: str) -> Optional[datetime]:
    """Parse the schema's timestamp format, tolerating a trailing Z."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def recency_from_last_seen(
    last_seen_at: str,
    *,
    now: Optional[datetime] = None,
    half_life_days: float = RECENCY_HALF_LIFE_DAYS,
) -> float:
    """Exponential decay on age, in [0, 1]. Newer scores higher.

        recency = 0.5 ** (age_days / half_life_days)

    Exponential rather than linear because linear needs an arbitrary cutoff age
    at which recency hits zero and everything older ties — and a fact from two
    years ago genuinely is less useful than one from one year ago, so the
    ordering should survive. Exponential decay never reaches zero, so old facts
    stay ranked relative to each other.

    A fact seen right now scores 1.0. One at exactly the half-life scores 0.5.
    A future timestamp (clock skew) is clamped to 1.0 rather than scoring above
    1.0 and breaking the [0, 1] contract the blend depends on.

    THE HALF-LIFE IS A PLACEHOLDER. See RECENCY_HALF_LIFE_DAYS.
    """
    parsed = _parse_iso(last_seen_at)
    if parsed is None:
        # An unparseable timestamp is treated as maximally stale rather than
        # raising: one malformed row should not fail an entire recall. It sorts
        # last on recency, which is the safe direction.
        return 0.0

    reference = now or datetime.now(timezone.utc)
    age_days = (reference - parsed).total_seconds() / 86400.0

    if age_days <= 0:
        return 1.0

    return 0.5 ** (age_days / half_life_days)


# ---------------------------------------------------------------------------
# the blend
# ---------------------------------------------------------------------------


def blended_score(relevance: float, recency: float) -> float:
    """Weighted sum. Both inputs are already in [0, 1], so the result is too.

    `importance` is not a factor — see IMPORTANCE_NOTE.
    """
    return (RELEVANCE_WEIGHT * relevance) + (RECENCY_WEIGHT * recency)


def score_candidate(
    distance: float,
    last_seen_at: str,
    *,
    now: Optional[datetime] = None,
) -> tuple[float, float, float]:
    """Returns (score, relevance, recency).

    All three are returned, not just the blend, because the components are the
    only way to answer "why did this rank here?" during Sprint 6/7 debugging.
    A single opaque number would make a bad ranking impossible to diagnose.
    """
    relevance = relevance_from_l2(distance)
    recency = recency_from_last_seen(last_seen_at, now=now)
    return blended_score(relevance, recency), relevance, recency
