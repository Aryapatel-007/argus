"""LLM judge for digests, and its calibration against hand-labelled cases.

The judge is NEVER the model that wrote the digest (ARGUS_BUILD 7.7). It
answers two yes/no questions about a digest given the ground-truth change
items: faithful (says nothing the items do not support) and complete
(mentions every change). Deterministic checks run alongside it; the judge
covers what they cannot (a wrong claim with no number in it).

Calibration: `CALIBRATION` holds hand-labelled digests, written with known
defects (invented number, a change called new, an omission, an unchanged
fact called changed). `calibrate()` reports each candidate judge's
agreement with the labels; the harness uses the judge with the best
agreement that is not the author.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from argus.graph.llm import LogContext
from argus.graph.llm_nodes import fence


class JudgeOutput(BaseModel):
    faithful: bool
    complete: bool
    reason: str


PROMPT = """You check a short monitoring digest against the true list of changes.

# True changes (ground truth)
<untrusted kind="facts">
{items}
</untrusted>

# Digest to check
<untrusted kind="digest">
{digest}
</untrusted>

# Questions
- faithful: true only if EVERY claim in the digest is supported by the true
  changes: no invented values, no wrong old/new values, no change claimed for
  something that did not change.
- complete: true only if the digest mentions EVERY true change.
- reason: one short sentence.

Return ONLY a JSON object: {{"faithful": true|false, "complete": true|false, "reason": "..."}}."""


def items_block(items: List[Dict[str, Any]]) -> str:
    lines = []
    for it in items:
        line = f"- [{it.get('change_kind')}] {it.get('entity_key')}: {fence(it.get('content'))}"
        if it.get("previous_content"):
            line += f" (previously: {fence(it['previous_content'])})"
        lines.append(line)
    return "\n".join(lines) or "(no changes)"


async def judge(backend, model: str, items: List[Dict[str, Any]], digest: str, log: LogContext) -> Optional[Dict]:
    msg = PROMPT.format(items=items_block(items), digest=fence(digest))
    res = await backend.chat(model=model, messages=[{"role": "user", "content": msg}], schema=JudgeOutput, log=log)
    return res.parsed


FEE = {"entity_key": "fees", "change_kind": "modified", "content": "The annual tuition fee for 2027 is AUD 48,500.",
       "previous_content": "The annual tuition fee for 2027 is AUD 45,000."}
INTAKE = {"entity_key": "intake_dates", "change_kind": "modified",
          "content": "Intakes: February, July and November each year.",
          "previous_content": "Intakes: February and July each year."}
SCHOL = {"entity_key": "scholarship", "change_kind": "added",
         "content": "A new merit scholarship covers 20% of tuition."}

#: (items, digest, faithful, complete). Labels written by hand.
CALIBRATION: List[Dict[str, Any]] = [
    {"items": [FEE], "digest": "The annual tuition fee rose to AUD 48,500 from AUD 45,000.", "faithful": True, "complete": True},
    {"items": [FEE], "digest": "The annual tuition fee rose to AUD 49,500 from AUD 45,000.", "faithful": False, "complete": True},
    {"items": [FEE], "digest": "Tuition is now AUD 48,500. The course was also shortened to one year.", "faithful": False, "complete": True},
    {"items": [FEE, INTAKE], "digest": "The tuition fee is now AUD 48,500, up from AUD 45,000.", "faithful": True, "complete": False},
    {"items": [FEE, INTAKE], "digest": "Fees rose to AUD 48,500 (was AUD 45,000), and a November intake was added.", "faithful": True, "complete": True},
    {"items": [INTAKE], "digest": "A November intake was added, so intakes are now February, July and November.", "faithful": True, "complete": True},
    {"items": [INTAKE], "digest": "The July intake was cancelled and replaced by November.", "faithful": False, "complete": True},
    {"items": [SCHOL], "digest": "A new merit scholarship now covers 20% of tuition.", "faithful": True, "complete": True},
    {"items": [SCHOL], "digest": "A new merit scholarship covers 50% of tuition.", "faithful": False, "complete": True},
    {"items": [SCHOL, FEE], "digest": "A new scholarship covers 20% of tuition.", "faithful": True, "complete": False},
    {"items": [FEE], "digest": "There is a new scholarship and the fee rose to AUD 48,500 from AUD 45,000.", "faithful": False, "complete": True},
    {"items": [SCHOL, FEE, INTAKE], "digest": "Fees went from AUD 45,000 to AUD 48,500, a November intake was added, "
                                              "and a merit scholarship covering 20% of tuition is new.", "faithful": True, "complete": True},
]


async def calibrate(backend, models: List[str], log: LogContext) -> Dict[str, Dict[str, Any]]:
    """Agreement of each candidate judge with the hand labels."""
    out = {}
    for m in models:
        agree_f = agree_c = n = invalid = 0
        for case in CALIBRATION:
            r = await judge(backend, m, case["items"], case["digest"], log)
            n += 1
            if r is None:
                invalid += 1
                continue
            agree_f += int(r["faithful"] == case["faithful"])
            agree_c += int(r["complete"] == case["complete"])
        out[m] = {"cases": n, "invalid": invalid, "faithful_agreement": agree_f / n,
                  "complete_agreement": agree_c / n, "agreement": (agree_f + agree_c) / (2 * n)}
    return out
