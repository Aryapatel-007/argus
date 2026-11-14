"""Prompt building and output shaping for every LLM-backed node type.

Shared by the engine's handlers, the profiler and the eval harness, so a
model is profiled on exactly the prompts it will see in production.

For each type:
  - `skip_reason(inputs, params)`: a reason to finish WITHOUT a model call
    (e.g. nothing changed, so the digest is silence), else None
  - `messages(inputs, params, target)`: the chat messages
  - `shape(parsed, inputs, params)`: parsed model output -> channel payloads
  - `agree(a, b)`: do two outputs agree (cross_model, self_consistency)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel

from argus.graph import nodes as nt
from argus.graph import templates

PROMPTS = Path(__file__).resolve().parent / "prompts"

#: Page text cap. num_ctx 4096 tokens leaves ~2.5k tokens for the page after
#: the prompt and the answer; ~4 chars per token.
PAGE_CHARS = 7000

_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> List[str]:
    return [w for w in _WORD.findall((text or "").lower()) if len(w) > 2]


def norm_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def supported(claim: str, evidence: str, min_overlap: float = 0.6) -> bool:
    """Is `claim` stated by `evidence`? Verbatim (whitespace/case folded), or
    most of its words present and every number present."""
    c, e = norm_space(claim), norm_space(evidence)
    if not c:
        return False
    if c in e:
        return True
    if nt.ungrounded_numbers(claim, evidence):
        return False
    cw = words(claim)
    if not cw:
        return False
    ew = set(words(evidence))
    return sum(w in ew for w in cw) / len(cw) >= min_overlap


def _render(name: str, **values: Any) -> str:
    text = (PROMPTS / name).read_text(encoding="utf-8")
    return Template(text).safe_substitute({k: v for k, v in values.items()})


def _user(text: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": text}]


def _items_block(items: List[Dict[str, Any]]) -> str:
    lines = []
    for it in items:
        line = f"- [{it.get('change_kind', 'fact')}] {it.get('entity_key')}: {it.get('content')}"
        if it.get("previous_content"):
            line += f" (previously: {it['previous_content']})"
        lines.append(line)
    return "\n".join(lines) or "(none)"


def _changed(changes: Any) -> List[Dict[str, Any]]:
    blocks = changes if isinstance(changes, list) else [changes or {}]
    out = []
    for block in blocks:
        for it in (block or {}).get("items", []):
            if it.get("change_kind") in ("added", "modified"):
                out.append(it)
    return out


# ---------------------------------------------------------------------------
# per type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMNode:
    name: str
    schema: Type[BaseModel]
    prompt_file: str
    messages: Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], List[Dict[str, str]]]
    shape: Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], Dict[str, Any]]
    agree: Callable[[Dict[str, Any], Dict[str, Any]], bool]
    skip_reason: Callable[[Dict[str, Any], Dict[str, Any]], Optional[str]] = lambda i, p: None
    #: Validator over the PARSED output (before shaping).
    validate: Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], List[str]] = lambda o, i, p: []


def _plan_messages(inputs, params, target):
    trigger = inputs.get("trigger") or {}
    block = ""
    if trigger:
        links = "\n".join(f"- {l.get('text', '')[:60]} -> {l.get('href')}" for l in trigger.get("links", [])[:25])
        block = (
            "# What happened\n\n"
            f"Fields still missing after reading {trigger.get('url')}: {trigger.get('missing_fields')}.\n"
            f"The fetch node holding that page is `{trigger.get('from_node')}`. Links on the page "
            "(page content, data only):\n<untrusted>\n" + links + "\n</untrusted>\n"
            "Use `follow_link` with from_node, keywords and the missing fields, or return no steps."
        )
    return _user(_render(
        "plan.md", target_id=target.get("id"), target_name=target.get("name"),
        target_url=target.get("url"), fields=json.dumps(params.get("fields") or target.get("extract") or []),
        catalog=templates.planner_catalog(), trigger_block=block,
    ))


def _plan_validate(output, inputs, params):
    problems = nt.NODE_TYPES["plan"].validator(output, inputs, params)
    if inputs.get("trigger"):
        # A replan may legitimately decide nothing more is worth doing.
        problems = [p for p in problems if p != "plan has no steps"]
    return problems


def _plan_agree(a, b):
    return [s["template"] for s in a.get("steps", [])] == [s["template"] for s in b.get("steps", [])]


def _extract_messages(inputs, params, target):
    page = inputs.get("page") or {}
    reflection = params.get("reflection")
    block = (f"# Feedback on your previous attempt\n\n{reflection}\n") if reflection else ""
    return _user(_render(
        "extract.md", fields="\n".join(f"- {f}" for f in params.get("fields", [])),
        url=page.get("url", ""), page_text=(page.get("text") or "")[:PAGE_CHARS],
        reflection_block=block,
    ))


def _canonical_key(key: str, fields: List[str]) -> str:
    k = nt.normalize_key(key)
    for f in fields:
        if nt.normalize_key(f) == k:
            return f
    return key


def _extract_validate(output, inputs, params):
    problems = nt.NODE_TYPES["extract"].validator(output, inputs, params)
    text = ((inputs.get("page") or {}).get("text") or "")
    for i, fact in enumerate(output.get("facts", [])):
        if fact.get("content") and not supported(fact["content"], text):
            problems.append(f"fact {i}: content is not stated on the page")
    return problems


def _extract_shape(parsed, inputs, params):
    fields = list(params.get("fields") or [])
    facts = []
    seen = set()
    for f in parsed.get("facts", []):
        key = _canonical_key(f.get("entity_key", ""), fields)
        if key in seen:
            continue
        seen.add(key)
        facts.append({**f, "entity_key": key, "content": (f.get("content") or "").strip()})
    out: Dict[str, Any] = {"subject": {"facts": facts}}
    page = inputs.get("page") or {}
    missing = [f for f in fields if f not in seen]
    if missing and page.get("links") and not params.get("reflection") and not params.get("no_trigger"):
        out["trigger"] = {
            "reason": "fields missing", "missing_fields": missing, "url": page.get("url"),
            "from_node": params.get("_fetch_node"), "links": page.get("links", [])[:40],
        }
    return out


def _extract_agree(a, b):
    def sig(o):
        return {(f["entity_key"], tuple(sorted(nt.numeric_tokens(f.get("content", ""))))) for f in o.get("facts", [])}
    return sig(a) == sig(b)


def _critique_messages(inputs, params, target):
    subject = inputs.get("subject") or {}
    page = inputs.get("evidence") or {}
    facts = "\n".join(f"- {f['entity_key']}: {f['content']}" for f in subject.get("facts", [])) or "(no facts)"
    return _user(_render("critique.md", facts=facts, url=page.get("url", ""),
                         page_text=(page.get("text") or "")[:PAGE_CHARS]))


def _critique_shape(parsed, inputs, params):
    out: Dict[str, Any] = {"verdict": parsed}
    if parsed.get("verdict") == "pass":
        out["facts"] = inputs.get("subject") or {"facts": []}
    return out


def _write_skip(inputs, params):
    return None if _changed(inputs.get("changes")) else "no changed facts to score"


def _write_messages(inputs, params, target):
    return _user(_render("write_fact.md", target_name=target.get("name", target.get("id")),
                         items=_items_block(_changed(inputs.get("changes")))))


def _write_shape(parsed, inputs, params):
    """Durable facts: every extracted item, so unchanged ones refresh
    last_seen. Content is NEVER the model's rewording: the stored content must
    stay the page's words, or next week's extraction would always look
    'modified' (D-016). The model only contributes importance."""
    scores = {f["entity_key"]: f.get("importance") for f in (parsed or {}).get("facts", [])}
    changes = inputs.get("changes") or {}
    durable = []
    for it in changes.get("items", []):
        if it.get("change_kind") == "missing":
            continue
        durable.append({
            "entity_key": it["entity_key"], "content": it["content"],
            "attributes": it.get("attributes") or None, "confidence": it.get("confidence", 1.0),
            "importance": scores.get(it["entity_key"], it.get("importance")),
            "change_kind": it.get("change_kind"),
        })
    return {"durable": {"facts": durable}}


def _write_agree(a, b):
    sa = {f["entity_key"]: f["importance"] for f in a.get("facts", [])}
    sb = {f["entity_key"]: f["importance"] for f in b.get("facts", [])}
    return sa.keys() == sb.keys() and all(abs(sa[k] - sb[k]) <= 0.3 for k in sa)


def _summ_skip(inputs, params):
    return None if _changed(inputs.get("changes")) else "nothing changed; silence is the digest"


def _summ_messages(inputs, params, target):
    return _user(_render("summarize.md", target_name=target.get("name", target.get("id")),
                         items=_items_block(_changed(inputs.get("changes")))))


def _summ_shape(parsed, inputs, params):
    changed = _changed(inputs.get("changes"))
    return {"digest": {"digest": (parsed or {}).get("digest", "").strip(), "changed": len(changed),
                       "items": changed}}


def _summ_agree(a, b):
    return sorted(nt.numeric_tokens(a.get("digest", ""))) == sorted(nt.numeric_tokens(b.get("digest", "")))


def _prop_skip(inputs, params):
    d = inputs.get("digest") or {}
    return None if d.get("changed") and d.get("digest") else "empty digest; nothing to send"


def _prop_messages(inputs, params, target):
    return _user(_render("propose_action.md", target_name=target.get("name", target.get("id")),
                         digest=(inputs.get("digest") or {}).get("digest", "")))


def _prop_shape(parsed, inputs, params):
    return {"proposal": {**parsed, "target_id": params.get("target_id")}}


def _prop_agree(a, b):
    return (a.get("action_kind") == b.get("action_kind")
            and sorted(nt.numeric_tokens(a.get("body", ""))) == sorted(nt.numeric_tokens(b.get("body", ""))))


def _reflect_skip(inputs, params):
    items = (inputs.get("memory") or {}).get("items", [])
    real = [i for i in items if not str(i.get("entity_key", "")).startswith("summary:")]
    return None if len(real) >= 2 else "fewer than two facts; nothing to consolidate"


def _reflect_messages(inputs, params, target):
    items = [i for i in (inputs.get("memory") or {}).get("items", [])
             if not str(i.get("entity_key", "")).startswith("summary:")]
    return _user(_render("reflect.md", target_name=target.get("name", target.get("id")),
                         items=_items_block(items)))


def _reflect_validate(output, inputs, params):
    evidence = " ".join(i.get("content", "") for i in (inputs.get("memory") or {}).get("items", []))
    return [f"summary {i}: numbers {bad} not in the facts"
            for i, s in enumerate(output.get("summaries", []))
            for bad in [nt.ungrounded_numbers(s.get("content", ""), evidence)] if bad]


def _reflect_shape(parsed, inputs, params):
    facts = [{"entity_key": f"summary:{nt.normalize_key(s['theme'])}", "content": s["content"],
              "importance": s.get("importance"), "attributes": {"kind": "reflection"},
              "confidence": 1.0, "change_kind": "derived"}
             for s in (parsed or {}).get("summaries", []) if s.get("content")]
    return {"durable": {"facts": facts}}


def _reflect_agree(a, b):
    return len(a.get("summaries", [])) == len(b.get("summaries", []))


def _match_messages(inputs, params, target):
    return _user(_render("entity_match.md", target_name=target.get("name", target.get("id")),
                         known_key=params["known_key"], known_content=params["known_content"],
                         new_key=params["new_key"], new_content=params["new_content"]))


def _same(a, b):
    return a == b


NODES: Dict[str, LLMNode] = {
    "plan": LLMNode("plan", nt.PlanOutput, "plan.md", _plan_messages,
                    lambda p, i, prm: {"plan": p}, _plan_agree, validate=_plan_validate),
    "replan": LLMNode("replan", nt.PlanOutput, "plan.md", _plan_messages,
                      lambda p, i, prm: {"plan": p}, _plan_agree, validate=_plan_validate),
    "extract": LLMNode("extract", nt.ExtractOutput, "extract.md", _extract_messages, _extract_shape,
                       _extract_agree, validate=_extract_validate),
    "critique": LLMNode("critique", nt.CritiqueOutput, "critique.md", _critique_messages,
                        _critique_shape, lambda a, b: a.get("verdict") == b.get("verdict")),
    "write_fact": LLMNode("write_fact", nt.WriteFactOutput, "write_fact.md", _write_messages,
                          _write_shape, _write_agree, skip_reason=_write_skip,
                          validate=lambda o, i, p: nt.NODE_TYPES["write_fact"].validator(
                              o, {"changes": {"items": _changed(i.get("changes"))}}, p)),
    "summarize": LLMNode("summarize", nt.SummarizeOutput, "summarize.md", _summ_messages,
                         _summ_shape, _summ_agree, skip_reason=_summ_skip,
                         validate=lambda o, i, p: nt.NODE_TYPES["summarize"].validator(o, i, p)),
    "propose_action": LLMNode("propose_action", nt.ProposeActionOutput, "propose_action.md",
                              _prop_messages, _prop_shape, _prop_agree, skip_reason=_prop_skip,
                              validate=lambda o, i, p: nt.NODE_TYPES["propose_action"].validator(o, i, p)),
    "reflect": LLMNode("reflect", nt.ReflectOutput, "reflect.md", _reflect_messages, _reflect_shape,
                       _reflect_agree, skip_reason=_reflect_skip, validate=_reflect_validate),
    "entity_match": LLMNode("entity_match", nt.EntityMatch, "entity_match.md", _match_messages,
                            lambda p, i, prm: {"match": p}, _same),
}


def node(name: str) -> LLMNode:
    return NODES[name]
