"""Part A: the larger embedding-prefix probe, deferred from Step 4b.

Step 4b compared three prefix forms on THREE generic sentence pairs and shipped
"no prefix" on that basis, explicitly flagging the sample as too small to
certify a retrieval strategy. This extends it to ten fact-shaped pairs — short
realistic fragments of the kind a browser-extraction tool actually produces
(job postings, program requirements, policy snippets), not general prose.

Each case is an (anchor, similar, unrelated) triple. The margin is
    cos(anchor, similar) - cos(anchor, unrelated)
so a POSITIVE margin means the embedding ranked the genuinely related fragment
above the unrelated one, which is the only property retrieval actually needs.
A negative margin is an inversion: the model preferred the wrong fragment.

Three forms, exactly as in Step 4b:
    none      - no prefix on either side
    doc/doc   - "search_document: " on both sides (symmetric)
    query/doc - "search_query: " on the anchor, "search_document: " on the
                candidates (asymmetric, the way nomic documents it)

Run: python scripts/probe_prefix.py
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from argus import embeddings  # noqa: E402

RUN_ID = "probe-prefix"

#: (label, anchor, similar, unrelated)
CASES = [
    (
        "job-title",
        "Senior Machine Learning Engineer",
        "Senior ML Engineer, applied research team",
        "Warehouse Operations Supervisor, night shift",
    ),
    (
        "salary",
        "Base salary AUD 165,000 plus superannuation",
        "Remuneration: $165k base plus super and bonus",
        "Applications close 30 November 2026",
    ),
    (
        "intake-dates",
        "Intake dates: February and July",
        "The program commences in Semester 1 (Feb) and Semester 2 (Jul)",
        "The campus library is open 24 hours during exam period",
    ),
    (
        "tuition",
        "Annual tuition fee is AUD 45,000 for international students",
        "International student fees: $45,000 per year",
        "Student clubs receive funding through the guild levy",
    ),
    (
        "entry-requirements",
        "Entry requires a bachelor degree with a WAM of 65 or above",
        "Admission: undergraduate degree, minimum weighted average mark 65",
        "Graduates work in data science, consulting and research roles",
    ),
    (
        "visa-policy",
        "Subclass 485 Temporary Graduate visa duration extended to 3 years",
        "Post-study work rights for the 485 visa now last three years",
        "Passport photographs must be taken within the last six months",
    ),
    (
        "duration",
        "Course duration is 2 years full time",
        "Full-time students complete the degree in two years",
        "Lectures are recorded and available on the learning portal",
    ),
    (
        "english-req",
        "IELTS overall 6.5 with no band below 6.0",
        "English requirement: IELTS 6.5, minimum 6.0 in each component",
        "The course is delivered at the Clayton and Caulfield campuses",
    ),
    (
        "location",
        "Position based in Sydney with hybrid work arrangement",
        "Sydney-based role, three days in office per week",
        "Reporting line is to the Head of Data Platform",
    ),
    (
        "skills",
        "Experience with Python, PyTorch and distributed training required",
        "Must know Python and PyTorch; large-scale model training experience",
        "Company offers 20 days annual leave and paid parental leave",
    ),
]

FORMS = {
    "none": ("", ""),
    "doc/doc": ("search_document: ", "search_document: "),
    "query/doc": ("search_query: ", "search_document: "),
}


def cosine(u, v) -> float:
    du = math.sqrt(sum(x * x for x in u))
    dv = math.sqrt(sum(x * x for x in v))
    return sum(a * b for a, b in zip(u, v)) / (du * dv)


async def main() -> None:
    print(f"model: {embeddings.model_tag()}")
    print(f"{len(CASES)} fact-shaped cases x {len(FORMS)} prefix forms\n")

    results = {name: [] for name in FORMS}

    header = f"{'case':<20}" + "".join(f"{name:>26}" for name in FORMS)
    print(header)
    print(f"{'':<20}" + "".join(f"{'sim   unrel   margin':>26}" for _ in FORMS))
    print("-" * len(header))

    for label, anchor, similar, unrelated in CASES:
        row = f"{label:<20}"
        for name, (anchor_prefix, candidate_prefix) in FORMS.items():
            anchor_vec = await embeddings.embed(anchor_prefix + anchor, run_id=RUN_ID)
            similar_vec = await embeddings.embed(candidate_prefix + similar, run_id=RUN_ID)
            unrelated_vec = await embeddings.embed(candidate_prefix + unrelated, run_id=RUN_ID)

            sim = cosine(anchor_vec, similar_vec)
            unrel = cosine(anchor_vec, unrelated_vec)
            margin = sim - unrel
            results[name].append(margin)
            row += f"{sim:>8.4f}{unrel:>8.4f}{margin:>+10.4f}"
        print(row)

    print("-" * len(header))
    summary = f"{'MEAN MARGIN':<20}"
    for name in FORMS:
        summary += f"{'':>16}{sum(results[name]) / len(results[name]):>+10.4f}"
    print(summary)

    inversions = f"{'INVERSIONS':<20}"
    for name in FORMS:
        count = sum(1 for m in results[name] if m < 0)
        inversions += f"{'':>16}{f'{count}/{len(CASES)}':>10}"
    print(inversions)

    worst = f"{'WORST MARGIN':<20}"
    for name in FORMS:
        worst += f"{'':>16}{min(results[name]):>+10.4f}"
    print(worst)

    print()
    ranked = sorted(FORMS, key=lambda n: sum(results[n]) / len(results[n]), reverse=True)
    best = ranked[0]
    clean = [n for n in FORMS if all(m > 0 for m in results[n])]
    print(f"highest mean margin : {best}")
    print(f"zero inversions     : {clean or 'none'}")


if __name__ == "__main__":
    asyncio.run(main())
