"""Sprint 2 probe: does think=False degrade the writer role specifically?

Deferred since Sprint 1. The writer is the one role that produces long free
prose (a digest paragraph), not short schema-constrained JSON — the one
shape where thinking might plausibly help, and where its absence degrades
quietly (a bland or incomplete digest) rather than failing validation loudly
the way a malformed planner/critic JSON does.

NOT production code. Does not import prompts/writer.md (that stub is a
single-fact-per-observation extractor, a different task shape from the
multi-fact digest this probe tests) and does not wire the writer role.
Purely a measurement script, mirroring scripts/probe_thinking.py's method.

Run: python scripts/probe_writer_thinking.py
Requires `ollama serve` running locally. Takes a while — 18 real
generations, some possibly long if thinking runs away on this prompt shape
too. Writes raw results incrementally (crash-safe) plus a progress line per
generation to stdout.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from string import Template

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx  # noqa: E402
from argus import config  # noqa: E402
from argus.ollama_client import get_client  # noqa: E402

MODEL = config.model_for_role("writer")  # qwen3.5:9b, from config - never hardcoded
KEEP_ALIVE = "30m"
RUNS_PER_VARIANT = 3

# Hard client-side timeout. This IS the kill mechanism for a runaway
# generation: ollama.Client.chat() is a blocking synchronous call with no
# separate cancel API, so a client-side read timeout is what turns "hung
# forever" into "recorded as a runaway and move on to the next generation".
# Sprint 1 measured a runaway at 1583s (26min) on a different, much shorter
# prompt; 300s (5min) here is the user-specified kill threshold for this run.
RUNAWAY_TIMEOUT_S = 300
SLOW_WARN_S = 90

RAW_RESULTS_PATH = Path(__file__).resolve().parent.parent / "notes" / "writer-probe-raw.json"

NS_PER_S = 1_000_000_000

# ---------------------------------------------------------------------------
# Probe-only digest prompt. NOT prompts/writer.md - see module docstring.
# string.Template ($var) to match house style (planner.md, writer.md).
# ---------------------------------------------------------------------------

DIGEST_PROMPT = Template("""You are drafting a weekly monitoring digest for one target.

# Target
$target_id

# Facts recalled for this target
$facts_block

# Task
Write a short digest paragraph (3-6 sentences) a human would read to know
what matters about this target this week.

# Rules
- State only what the facts above actually show. Do not infer a change, a
  number, or a date that is not written in one of the facts.
- Each fact above is marked added, modified, or unchanged. Reflect that
  status accurately - do not call an unchanged fact new, and do not omit a
  real change.
- Plain prose. No headers, no bullet list, no JSON, no meta-commentary about
  this being a digest or a monitoring run.
""")


def format_facts_block(facts: list) -> str:
    lines = []
    for f in facts:
        status_line = f"status: {f['status']}"
        if f.get("prior_content"):
            status_line += f" (previously: \"{f['prior_content']}\")"
        lines.append(
            f"- entity: {f['entity_key']}\n"
            f"  content: \"{f['content']}\"\n"
            f"  {status_line}\n"
            f"  source: {f.get('source_url') or 'n/a'}\n"
            f"  relevance={f['relevance']:.2f} recency={f['recency']:.2f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fact sets. Shape matches RecalledFact's real fields (fact_id, target_id,
# entity_key, content, attributes, score, relevance, recency, last_seen_at,
# source_url) plus two probe-only additions - "status" and "prior_content" -
# representing what a real digest job would ALSO know from list_changes_since
# or from remember_fact's own RememberResult during the run. RecalledFact
# itself carries no `confidence` field (confirmed against memory.py), so
# Set C's "low confidence" is expressed through relevance/recency and content
# phrasing, not a literal confidence number - see notes/writer-probe-results.md.
# ---------------------------------------------------------------------------

FACT_SETS = {
    "A_no_changes": {
        "target_id": "monash_ai_masters",
        "description": "No changes since last run - all facts unchanged.",
        "facts": [
            dict(fact_id=1, entity_key="intake_dates",
                 content="Intake dates: February and July.",
                 status="unchanged", prior_content=None,
                 source_url="https://www.monash.edu/study/courses/find-a-course/artificial-intelligence-c6001",
                 relevance=0.81, recency=0.94, last_seen_at="2026-08-24T00:00:00.000Z"),
            dict(fact_id=2, entity_key="duration",
                 content="Course duration is 2 years full time.",
                 status="unchanged", prior_content=None,
                 source_url="https://www.monash.edu/study/courses/find-a-course/artificial-intelligence-c6001",
                 relevance=0.77, recency=0.94, last_seen_at="2026-08-24T00:00:00.000Z"),
            dict(fact_id=3, entity_key="tuition",
                 content="Annual tuition fee is AUD 45,000 for domestic and international students.",
                 status="unchanged", prior_content=None,
                 source_url="https://www.monash.edu/study/courses/find-a-course/artificial-intelligence-c6001",
                 relevance=0.83, recency=0.94, last_seen_at="2026-08-24T00:00:00.000Z"),
            dict(fact_id=4, entity_key="entry_requirements",
                 content="Entry requires a bachelor degree with a WAM of 65 or above.",
                 status="unchanged", prior_content=None,
                 source_url="https://www.monash.edu/study/courses/find-a-course/artificial-intelligence-c6001",
                 relevance=0.74, recency=0.94, last_seen_at="2026-08-24T00:00:00.000Z"),
            dict(fact_id=5, entity_key="english_requirement",
                 content="IELTS overall 6.5 with no band below 6.0.",
                 status="unchanged", prior_content=None,
                 source_url="https://www.monash.edu/study/courses/find-a-course/artificial-intelligence-c6001",
                 relevance=0.69, recency=0.94, last_seen_at="2026-08-24T00:00:00.000Z"),
        ],
    },
    "B_genuine_changes": {
        "target_id": "rmit_ai_masters",
        "description": "2-3 genuine changes: one added, two modified, two unchanged.",
        "facts": [
            dict(fact_id=11, entity_key="tuition",
                 content="Annual tuition fee is AUD 48,500 for the 2027 intake.",
                 status="modified",
                 prior_content="Annual tuition fee is AUD 44,000 for the 2026 intake.",
                 source_url="https://www.rmit.edu.au/study-with-us/levels-of-study/postgraduate-study/masters-by-coursework/master-of-artificial-intelligence-mc267",
                 relevance=0.85, recency=0.97, last_seen_at="2026-08-25T00:00:00.000Z"),
            dict(fact_id=12, entity_key="intake_dates",
                 content="Intake dates: March, July and November.",
                 status="modified",
                 prior_content="Intake dates: March and July.",
                 source_url="https://www.rmit.edu.au/study-with-us/levels-of-study/postgraduate-study/masters-by-coursework/master-of-artificial-intelligence-mc267",
                 relevance=0.80, recency=0.97, last_seen_at="2026-08-25T00:00:00.000Z"),
            dict(fact_id=13, entity_key="scholarship",
                 content="New RMIT AI Excellence Scholarship covers 25% of tuition for the first 20 applicants each intake.",
                 status="added", prior_content=None,
                 source_url="https://www.rmit.edu.au/study-with-us/levels-of-study/postgraduate-study/masters-by-coursework/master-of-artificial-intelligence-mc267",
                 relevance=0.72, recency=0.97, last_seen_at="2026-08-25T00:00:00.000Z"),
            dict(fact_id=14, entity_key="duration",
                 content="Course duration is 2 years full time.",
                 status="unchanged", prior_content=None,
                 source_url="https://www.rmit.edu.au/study-with-us/levels-of-study/postgraduate-study/masters-by-coursework/master-of-artificial-intelligence-mc267",
                 relevance=0.68, recency=0.90, last_seen_at="2026-08-24T00:00:00.000Z"),
            dict(fact_id=15, entity_key="entry_requirements",
                 content="Entry requires a bachelor degree in a related field with a credit average.",
                 status="unchanged", prior_content=None,
                 source_url="https://www.rmit.edu.au/study-with-us/levels-of-study/postgraduate-study/masters-by-coursework/master-of-artificial-intelligence-mc267",
                 relevance=0.65, recency=0.90, last_seen_at="2026-08-24T00:00:00.000Z"),
        ],
    },
    "C_ambiguous_thin": {
        "target_id": "homeaffairs_skilled_visa",
        "description": (
            "Ambiguous/thin data: one contradictory-looking pair of facts "
            "(485 visa duration), lower relevance throughout, one vague fact."
        ),
        "facts": [
            dict(fact_id=21, entity_key="visa_485_duration",
                 content="Subclass 485 Temporary Graduate visa duration extended to 3 years for eligible graduates.",
                 status="added", prior_content=None,
                 source_url="https://immi.homeaffairs.gov.au/what-we-do/skilled-migration-program",
                 relevance=0.58, recency=0.99, last_seen_at="2026-08-25T00:00:00.000Z"),
            dict(fact_id=22, entity_key="visa_485_duration_regional_note",
                 content="Subclass 485 visa remains at 2 years for non-regional graduates; only regional-area graduates receive the 3-year extension.",
                 status="added", prior_content=None,
                 source_url="https://immi.homeaffairs.gov.au/what-we-do/skilled-migration-program",
                 relevance=0.55, recency=0.99, last_seen_at="2026-08-25T00:00:00.000Z"),
            dict(fact_id=23, entity_key="points_test_note",
                 content="Skilled migration points test changes were mentioned in passing; the page does not state the new thresholds.",
                 status="unchanged", prior_content=None,
                 source_url="https://immi.homeaffairs.gov.au/what-we-do/skilled-migration-program",
                 relevance=0.41, recency=0.85, last_seen_at="2026-08-20T00:00:00.000Z"),
            dict(fact_id=24, entity_key="announcement_date",
                 content="Announcement dated 3 August 2026.",
                 status="unchanged", prior_content=None,
                 source_url="https://immi.homeaffairs.gov.au/what-we-do/skilled-migration-program",
                 relevance=0.44, recency=0.85, last_seen_at="2026-08-20T00:00:00.000Z"),
        ],
    },
}


def build_prompt(set_key: str) -> str:
    fset = FACT_SETS[set_key]
    return DIGEST_PROMPT.substitute(
        target_id=fset["target_id"],
        facts_block=format_facts_block(fset["facts"]),
    )


def call(client, prompt: str, think: bool) -> dict:
    """One generation. Returns a result dict; never raises - timeouts and
    other errors are captured and returned as data, per the requirement to
    continue the probe rather than let one generation hang the rest.
    """
    start = time.perf_counter()
    try:
        response = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            think=think,
            keep_alive=KEEP_ALIVE,
            options={"temperature": 0},
        )
    except httpx.TimeoutException as exc:
        elapsed = time.perf_counter() - start
        return {
            "ok": False,
            "runaway": True,
            "elapsed_s": elapsed,
            "error": f"{type(exc).__name__}: {exc}",
            "content": None,
            "thinking": None,
            "eval_count": None,
            "prompt_eval_count": None,
            "total_duration_s": None,
        }
    except Exception as exc:  # noqa: BLE001 - must be captured, not fatal to the probe
        elapsed = time.perf_counter() - start
        return {
            "ok": False,
            "runaway": False,
            "elapsed_s": elapsed,
            "error": f"{type(exc).__name__}: {exc}",
            "content": None,
            "thinking": None,
            "eval_count": None,
            "prompt_eval_count": None,
            "total_duration_s": None,
        }

    elapsed = time.perf_counter() - start
    message = response.message
    return {
        "ok": True,
        "runaway": False,
        "elapsed_s": elapsed,
        "error": None,
        "content": message.content,
        "thinking": getattr(message, "thinking", None),
        "eval_count": response.eval_count,
        "prompt_eval_count": response.prompt_eval_count,
        "eval_duration_s": (response.eval_duration / NS_PER_S) if response.eval_duration else None,
        "total_duration_s": (response.total_duration / NS_PER_S) if response.total_duration else None,
    }


def main() -> None:
    print(f"model: {MODEL}")
    print(f"runaway kill threshold: {RUNAWAY_TIMEOUT_S}s, slow-warn threshold: {SLOW_WARN_S}s")
    print()

    # Separate client with a generous timeout - this probe's think=True calls
    # on a long free-prose task may legitimately run well past the app's
    # normal 120s TIMEOUT_SECONDS. The 300s ceiling here IS the runaway kill.
    client = get_client(config.ollama_options().get("host"), RUNAWAY_TIMEOUT_S)

    results: dict = {}
    total_start = time.perf_counter()

    for set_key in FACT_SETS:
        results[set_key] = {}
        prompt = build_prompt(set_key)
        for think in (False, True):
            variant = f"think_{think}"
            results[set_key][variant] = []
            for run_number in range(1, RUNS_PER_VARIANT + 1):
                label = f"{set_key} / think={think} / run {run_number}"
                print(f"[{label}] starting...", flush=True)

                result = call(client, prompt, think)
                results[set_key][variant].append(result)

                # Persist incrementally - crash-safe, and lets partial
                # results be inspected without waiting for the whole run.
                RAW_RESULTS_PATH.write_text(
                    json.dumps(results, indent=2), encoding="utf-8"
                )

                if result["runaway"]:
                    print(
                        f"[{label}] RUNAWAY - killed after {result['elapsed_s']:.1f}s "
                        f"(>{RUNAWAY_TIMEOUT_S}s ceiling): {result['error']}",
                        flush=True,
                    )
                elif not result["ok"]:
                    print(f"[{label}] ERROR after {result['elapsed_s']:.1f}s: {result['error']}", flush=True)
                else:
                    slow_marker = " *** SLOW (>90s) ***" if result["elapsed_s"] > SLOW_WARN_S else ""
                    tok_info = ""
                    if think:
                        thinking_len = len(result["thinking"] or "")
                        content_len = len(result["content"] or "")
                        tok_info = (
                            f" eval_count={result['eval_count']} "
                            f"thinking_chars={thinking_len} content_chars={content_len}"
                        )
                    print(
                        f"[{label}] ok, {result['elapsed_s']:.1f}s{tok_info}{slow_marker}",
                        flush=True,
                    )

    total_elapsed = time.perf_counter() - total_start
    print()
    print(f"TOTAL WALL CLOCK: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"raw results written to: {RAW_RESULTS_PATH}")


if __name__ == "__main__":
    main()
