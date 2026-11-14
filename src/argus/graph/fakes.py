"""A deterministic stand-in for Ollama, for tests and crash tests.

It answers each node type the way a careful model would, from the prompt
alone, and writes real llm_calls rows through the same log path as the real
backend, so trajectory assertions see exactly what production would log.

Knobs:
  fail_first[(node_role, model)] = n   first n calls return invalid JSON
  wrong_number_first[model] = n        first n extract calls invent a fee
  delay_s                              await this long inside every call
  hang_on                              (role, call_index) -> sleep forever
                                       (crash tests kill the process there)
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from argus.graph import nodes as nt
from argus.graph.llm import CallResult, LogContext, _log, parse

KEYWORDS = {
    "fees": ["fee", "tuition", "aud"], "duration": ["duration", "years", "year"],
    "intake_dates": ["intake"], "entry_requirements": ["entry", "wam", "bachelor"],
    "announcement_title": ["announcement", "visa"], "announcement_date": ["published", "dated"],
    "summary": ["from", "will"], "scholarship": ["scholarship"],
}


def unit_vector(text: str, dims: int = 768) -> List[float]:
    """Deterministic unit vector for `text` (sha256-seeded). Same text, same
    vector; different texts land far apart. Stands in for nomic-embed-text."""
    import hashlib
    import math

    raw = hashlib.sha256(text.encode("utf-8")).digest() * (dims // 32 + 1)
    vec = [(b - 127.5) for b in raw[:dims]]
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


def install_fake_embeddings() -> None:
    """Replace argus.embeddings.embed with unit_vector, process-wide."""
    from argus import embeddings

    async def embed(text, *, run_id, conn=None, task_id=None, step_index=None):
        if isinstance(text, str):
            return unit_vector(text)
        return [unit_vector(t) for t in text]

    embeddings.embed = embed


class _Resp:
    def __init__(self, content: str, eval_count: int):
        self.message = type("M", (), {"content": content, "thinking": None})()
        self.eval_count = eval_count
        self.prompt_eval_count = 200
        self.eval_duration = int(eval_count / 40 * 1e9)
        self.total_duration = self.eval_duration + 100_000_000
        self.load_duration = 0


def _between(text: str, start: str, end: str) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end, i + len(start))
    return text[i + len(start): j if j >= 0 else None]


def _fence(text: str, marker: str) -> str:
    """Contents of the first <untrusted ...> block whose opening tag contains marker."""
    # Require an attribute: the prompts also MENTION "<untrusted>" in prose.
    for m in re.finditer(r"<untrusted(\s[^>]*)>(.*?)</untrusted>", text, re.S):
        if marker in m.group(1):
            return m.group(2)
    return ""


def _sentences(text: str) -> List[str]:
    # Line breaks end a sentence too (page text has one block per line).
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


class FakeBackend:
    def __init__(self, *, delay_s: float = 0.0):
        self.delay_s = delay_s
        self.calls: List[Tuple[str, str]] = []
        self.counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self.fail_first: Dict[Tuple[str, str], int] = {}
        self.wrong_number_first: Dict[str, int] = {}
        #: the first n extract calls OVERALL (any model) put the fee sentence
        #: under `duration`. Only a critic can catch this: every word and
        #: number IS on the page. Counted across models on purpose: the router
        #: may send a retry to a different model.
        self.mislabel_first: int = 0
        self.hang_on: Optional[Tuple[str, int]] = None
        self._resident: List[str] = []

    async def resident_models(self) -> List[str]:
        return list(self._resident)

    async def chat(self, *, model, messages, schema, log: LogContext, temperature=0.0, seed=None) -> CallResult:
        role = log.role
        key = (role, model)
        self.counts[key] += 1
        index = sum(1 for r, _ in self.calls if r == role)
        self.calls.append((role, model))
        self._resident = [model]
        if self.hang_on == (role, index):
            await asyncio.sleep(3600)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        prompt = messages[-1]["content"]
        if self.counts[key] <= self.fail_first.get(key, 0):
            raw = "{not json"
        else:
            raw = json.dumps(self._answer(role, model, prompt))
        resp = _Resp(raw, 60 + len(raw) // 4)
        parsed, error = parse(schema, raw)
        _log(log, model, resp, parsed is not None, False, error, raw)
        return CallResult(model_tag=model, parsed=parsed, raw=raw, error=error,
                          eval_count=resp.eval_count, prompt_eval_count=200, tok_s=40.0, wall_s=0.01)

    # ---------------------------------------------------------------- brain

    def _answer(self, role: str, model: str, prompt: str) -> Dict[str, Any]:
        if role in ("plan", "replan"):
            url = _between(prompt, "- url: ", "\n").strip()
            fields = json.loads(_between(prompt, "- fields to find: ", "\n").strip() or "[]")
            if role == "replan":
                frm = re.search(r"fetch node holding that page is `([^`]+)`", prompt)
                if frm:
                    return {"thought": "follow the link", "steps": [{"template": "follow_link", "params": {
                        "from_node": frm.group(1), "keywords": ["fee", "scholarship"], "fields": fields}}]}
                return {"thought": "nothing more", "steps": []}
            return {"thought": "check the page", "steps": [{"template": "check_page",
                                                            "params": {"url": url, "fields": fields}}]}
        if role == "extract":
            page = _fence(prompt, "source=")
            fields = [l[2:].strip() for l in _between(prompt, "# Fields to find", "# Page").splitlines()
                      if l.startswith("- ")]
            facts = []
            wrong = self.wrong_number_first.get(model, 0)
            for f in fields:
                for s in _sentences(page):
                    low = s.lower()
                    if "ignore" in low:
                        continue
                    if any(k in low for k in KEYWORDS.get(f, [f.replace("_", " ")])):
                        content = s
                        if f == "fees" and wrong > 0:
                            content = re.sub(r"\d[\d,]*", "99,999", s, count=1)
                        facts.append({"entity_key": f, "content": content, "attributes": {}, "confidence": 0.9})
                        break
            if wrong > 0:
                self.wrong_number_first[model] = wrong - 1
            if self.mislabel_first > 0:
                self.mislabel_first -= 1
                by_key = {f["entity_key"]: f for f in facts}
                if "fees" in by_key and "duration" in by_key:
                    by_key["duration"]["content"] = by_key["fees"]["content"]
            return {"facts": facts}
        if role == "critique":
            page = _fence(prompt, "source=")
            facts = _fence(prompt, 'kind="facts"')
            bad = nt.ungrounded_numbers(facts, page)
            mislabeled = [k for k, c in re.findall(r"- ([\w:]+): (.*)", facts)
                          if k in KEYWORDS and not any(w in c.lower() for w in KEYWORDS[k])]
            if bad or mislabeled:
                why = (f"numbers {bad} are not on the page" if bad else
                       f"{mislabeled} hold text about a different field")
                return {"verdict": "fail", "confidence": 0.9, "reflection": f"{why}; quote the right sentence"}
            return {"verdict": "pass", "confidence": 0.9, "reflection": ""}
        if role == "write_fact":
            items = re.findall(r"- \[(\w+)\] ([\w:]+): (.*?)(?: \(previously:.*)?$", prompt, re.M)
            return {"facts": [{"entity_key": k, "content": c, "importance": 0.8} for _, k, c in items]}
        if role == "summarize":
            items = re.findall(r"- \[(\w+)\] ([\w:]+): (.*)$", prompt, re.M)
            parts = []
            for kind, key, rest in items:
                content = rest.split(" (previously:")[0]
                prev = re.search(r"\(previously: (.*)\)$", rest)
                parts.append(f"{key.replace('_', ' ')} {('changed to ' if kind == 'modified' else 'is new: ')}"
                             f"{content}" + (f" (was: {prev.group(1)})" if prev else ""))
            return {"digest": ". ".join(parts) + "."}
        if role == "propose_action":
            digest = _fence(prompt, 'kind="digest"').strip()
            return {"action_kind": "outbox_message", "subject": "Changes detected",
                    "body": digest[:2000], "reasoning": "the digest reports real changes"}
        if role == "reflect":
            items = re.findall(r"- \[[\w]+\] ([\w:]+): (.*)$", prompt, re.M)
            return {"summaries": [{"theme": "overview", "content": " ".join(c for _, c in items[:2]),
                                   "importance": 0.5}]}
        if role == "resolve_entity":
            a = _between(prompt, "entity_key: ", "\n")
            b = prompt.rsplit("entity_key: ", 1)[-1].split("\n")[0]
            same = bool(set(a.split("_")) & set(b.split("_")))
            return {"same_entity": same, "confidence": 0.8}
        return {}
