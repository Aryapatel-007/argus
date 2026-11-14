"""Node handlers: what each node type does when it runs.

A handler never writes graph state. It returns a NodeResult; the engine
stores it (status, output, event, extra ops) in one transaction. Handlers
never raise to the engine for an expected failure: a failed page, an invalid
model answer or a rejected critique is a NodeResult with status 'failed', so
one bad page degrades a run instead of killing it.

Side effects:
  - approval_gate writes ONE approvals row (UNIQUE idempotency key, stored
    with the node before it ran) via extra_ops, in the same transaction as
    the gate's running -> blocked.
  - execute_action re-reads that row and acts only if it says 'approved'
    (third, independent check after the validator and readiness). The effect
    is idempotent by key; the status flip to 'executed' is conditional on
    'approved' and lands in the same transaction as the node's done.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

from argus.graph import llm_nodes
from argus.graph import nodes as nt
from argus.graph.strategies import StrategyRunner

Op = Tuple[str, Tuple[Any, ...]]


@dataclass
class NodeResult:
    status: str  # done | failed | blocked | skipped
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    skip_reason: Optional[str] = None
    tokens: int = 0
    extra_ops: List[Op] = field(default_factory=list)
    #: (node_type, model, source, reward, detail, key_suffix). The suffix keeps
    #: rewards for different samples of one node distinct (e.g. "#2").
    rewards: List[Tuple[str, str, str, float, str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


GENERIC_LLM = {"plan", "replan", "extract", "critique", "write_fact", "summarize",
               "propose_action", "reflect"}


async def run(ex: Any) -> NodeResult:
    t = ex.node.node_type
    if t in GENERIC_LLM:
        return await _llm(ex)
    handler = {
        "resolve_entity": _resolve, "diff": _diff, "recall": _recall, "remember": _remember,
        "fetch_page": _fetch, "search": _search, "approval_gate": _gate,
        "execute_action": _execute,
    }.get(t)
    if handler is None:
        return NodeResult("failed", error=f"no handler for node type {t!r}")
    return await handler(ex)


# ---------------------------------------------------------------------------
# LLM nodes
# ---------------------------------------------------------------------------


def _validator_rewards(node_type: str, res) -> List[Tuple[str, str, str, float, str]]:
    out = []
    for a in res.attempts:
        if a.reused:
            continue
        out.append((node_type, a.model, "validator", 1.0 if a.valid else 0.0,
                    "; ".join(a.problems)[:300] if a.problems else "valid", f"#{a.index}"))
    return out


async def _llm(ex: Any) -> NodeResult:
    t = ex.node.node_type
    spec = llm_nodes.node(t)
    reason = spec.skip_reason(ex.inputs, ex.params)
    if reason:
        if t == "propose_action":
            return NodeResult("skipped", skip_reason=reason)
        return NodeResult("done", output=spec.shape({}, ex.inputs, ex.params), notes=[reason])
    if ex.decision is None:
        return NodeResult("failed", error="no routing decision stored for an LLM node")
    res = await StrategyRunner(ex=ex, spec=spec, decision=ex.decision).run()
    rewards = _validator_rewards(t, res)
    if not res.ok:
        return NodeResult("failed", error=res.error or "model output rejected", tokens=res.tokens,
                          rewards=rewards, notes=res.notes)
    shaped = spec.shape(res.parsed, ex.inputs, ex.params)
    # Which model wrote the accepted answer. The critic independence rule and
    # the critic reward use THIS, not every model the decision named (a
    # cascade names two but only one authored the output; review High 2).
    shaped["_produced_by"] = res.model
    if t == "critique" and res.parsed.get("verdict") == "fail":
        return NodeResult("failed", output=shaped, tokens=res.tokens, rewards=rewards,
                          error=f"critic rejected: {res.parsed.get('reflection', '')}"[:800],
                          notes=res.notes)
    if res.uncertainty is not None:
        shaped = {**shaped, "_uncertainty": res.uncertainty}
    return NodeResult("done", output=shaped, tokens=res.tokens, rewards=rewards, notes=res.notes)


# ---------------------------------------------------------------------------
# resolve_entity: normalized key -> embedding similarity -> model check
# ---------------------------------------------------------------------------

def _resolve_cfg(ex: Any) -> Dict[str, float]:
    cfg = (getattr(ex, "routing_cfg", None) or {}).get("resolve", {})
    return {"same": float(cfg.get("embed_same", 0.85)), "maybe": float(cfg.get("embed_maybe", 0.60)),
            "unchanged_overlap": float(cfg.get("unchanged_overlap", 0.85))}


def _pair_index(a: str, b: str) -> int:
    """Sample index derived from the PAIR being compared, so a resumed node
    that sees a different top candidate cannot reuse an answer about another
    pair (review low)."""
    import hashlib

    return 100 + int(hashlib.sha256(f"{a}\x00{b}".encode()).hexdigest()[:8], 16) % 1_000_000


async def _resolve(ex: Any) -> NodeResult:
    facts = ((ex.inputs.get("facts") or {}).get("facts")) or []
    memory_items = ((ex.inputs.get("memory") or {}).get("items")) or []
    known = {m["entity_key"]: m for m in memory_items}
    known_norm = {nt.normalize_key(k): k for k in known}
    claimed = set()
    out, tokens, rewards = [], 0, []
    spec = llm_nodes.node("entity_match")
    th = _resolve_cfg(ex)
    for i, fact in enumerate(facts):
        key = fact["entity_key"]
        nk = nt.normalize_key(key)
        path, detail, resolved = None, None, key
        if key in known:
            path = "exact"
        elif nk in known_norm:
            path, resolved = "normalized", known_norm[nk]
        elif memory_items:
            hits = await ex.services.memory.recall(fact["content"], ex.target["id"], 3)
            hits = [h for h in hits if h["entity_key"] not in claimed
                    and not str(h["entity_key"]).startswith("summary:")]
            top = hits[0] if hits else None
            rel = float(top["relevance"]) if top else 0.0
            if top and rel >= th["same"]:
                path, resolved, detail = "embedding", top["entity_key"], {"relevance": rel}
            elif top and rel >= th["maybe"] and ex.decision is not None:
                params = {"known_key": top["entity_key"], "known_content": top["content"],
                          "new_key": key, "new_content": fact["content"]}
                runner = StrategyRunner(ex=ex, spec=spec, decision=ex.decision)
                a = await runner.call(_pair_index(top["entity_key"] + "|" + top["content"],
                                                  key + "|" + fact["content"]),
                                      ex.decision["model_tag"], inputs={}, params=params)
                tokens += a.eval_count
                rewards.append(("resolve_entity", a.model, "validator", 1.0 if a.valid else 0.0,
                                "entity_match", f"#{a.index}"))
                same = bool(a.valid and a.parsed.get("same_entity"))
                path = "model"
                detail = {"relevance": rel, "same_entity": same, "model": a.model}
                if same:
                    resolved = top["entity_key"]
            else:
                path, detail = "new", {"relevance": rel}
        else:
            path = "new"
        claimed.add(resolved)
        out.append({**fact, "entity_key": resolved, "extracted_key": key,
                    "resolution": path, "resolution_detail": detail})
    return NodeResult("done", output={"resolved": {"items": out}}, tokens=tokens, rewards=rewards)


# ---------------------------------------------------------------------------
# diff: grounding-based, so extraction wording drift is not a "change" (D-016)
# ---------------------------------------------------------------------------


async def _diff(ex: Any) -> NodeResult:
    resolved = ((ex.inputs.get("resolved") or {}).get("items")) or []
    memory_items = {m["entity_key"]: m for m in ((ex.inputs.get("memory") or {}).get("items")) or []}
    text = ((ex.inputs.get("evidence") or {}).get("text")) or ""
    fields = list(ex.params.get("fields") or [])
    page = ex.inputs.get("evidence") or {}
    # The page actually read, not a template param: a page reached by
    # follow_link has no url in its params (it came from a search node).
    source_url = page.get("final_url") or page.get("url") or ex.params.get("source_url")
    items, seen = [], set()
    for r in resolved:
        key = r["entity_key"]
        seen.add(key)
        stored = memory_items.get(key)
        base = {"entity_key": key, "attributes": r.get("attributes") or None,
                "confidence": r.get("confidence", 1.0), "resolution": r.get("resolution"),
                "source_url": source_url}
        if stored is None:
            items.append({**base, "content": r["content"], "change_kind": "added"})
        elif (llm_nodes.norm_space(stored["content"]) == llm_nodes.norm_space(r["content"])
              or llm_nodes.supported(stored["content"], text, min_overlap=_resolve_cfg(ex)["unchanged_overlap"])):
            items.append({**base, "content": stored["content"], "change_kind": "unchanged",
                          "attributes": stored.get("attributes"), "importance": stored.get("importance")})
        else:
            items.append({**base, "content": r["content"], "change_kind": "modified",
                          "previous_content": stored["content"]})
    for key in fields:
        if key in seen or key not in memory_items:
            continue
        stored = memory_items[key]
        if llm_nodes.supported(stored["content"], text, min_overlap=_resolve_cfg(ex)["unchanged_overlap"]):
            items.append({"entity_key": key, "content": stored["content"], "change_kind": "unchanged",
                          "attributes": stored.get("attributes"), "importance": stored.get("importance"),
                          "resolution": "still_on_page"})
        else:
            items.append({"entity_key": key, "content": stored["content"], "change_kind": "missing"})
    return NodeResult("done", output={"changes": {
        "target_id": ex.target["id"], "source_url": source_url, "items": items}})


# ---------------------------------------------------------------------------
# memory and browser tools
# ---------------------------------------------------------------------------


async def _recall(ex: Any) -> NodeResult:
    items = await ex.services.memory.list_current_facts(ex.params.get("target_id") or ex.target["id"])
    return NodeResult("done", output={"memory": {"items": items}})


async def _remember(ex: Any) -> NodeResult:
    facts = ((ex.inputs.get("durable") or {}).get("facts")) or []
    results = []
    for f in facts:
        r = await ex.services.memory.remember(
            target_id=ex.target["id"], entity_key=f["entity_key"], content=f["content"],
            attributes=f.get("attributes") or None,
            source_url=f.get("source_url") or ex.params.get("source_url") or None,
            confidence=float(f.get("confidence") if f.get("confidence") is not None else 1.0),
            importance=f.get("importance"),
            # Same node + same entity -> same key: a resumed remember returns the
            # first write's result and writes nothing (audit H3).
            idempotency_key=f"{ex.node.node_id}:{f['entity_key']}",
        )
        results.append({"entity_key": f["entity_key"], **r})
    return NodeResult("done", output={"stored": {"results": results}})


def same_site(url: str, base: str) -> bool:
    """Same host (ignoring a leading www.) AND same port. Fails closed: no
    base url, no match (review low: a missing target url used to make every
    url "same site")."""
    if not url or not base:
        return False
    a, b = urlparse(url), urlparse(base)
    ha, hb = (a.hostname or "").lower(), (b.hostname or "").lower()
    strip = lambda h: h[4:] if h.startswith("www.") else h  # noqa: E731
    default = {"http": 80, "https": 443}
    pa = a.port or default.get(a.scheme)
    pb = b.port or default.get(b.scheme)
    return a.scheme in ("http", "https") and bool(ha) and strip(ha) == strip(hb) and pa == pb


async def _fetch(ex: Any) -> NodeResult:
    urls = [u for u in (ex.inputs.get("url") or []) if u]
    url = urls[0] if urls else ex.params.get("url")
    if not url:
        return NodeResult("failed", error="no url to fetch")
    if not same_site(url, ex.target.get("url", "")):
        return NodeResult("failed", error=f"refusing off-site url {url!r} for target {ex.target['id']}")
    page = await ex.services.browser.open_page(url)
    if not page.get("ok") or not (page.get("text") or "").strip():
        return NodeResult("failed", error=f"fetch failed: status={page.get('status')} {page.get('error')}")
    return NodeResult("done", output={"page": page, "evidence": page},
                      notes=[f"partial={page.get('partial')} confidence={page.get('confidence')}"])


async def _search(ex: Any) -> NodeResult:
    page = ex.inputs.get("page") or {}
    keywords = [k.lower() for k in ex.params.get("keywords") or []]
    best, best_score = None, 0
    for link in page.get("links", []):
        href = urljoin(page.get("final_url") or page.get("url") or "", link.get("href", ""))
        if not same_site(href, ex.target.get("url", "")):
            continue
        hay = f"{link.get('text', '')} {href}".lower()
        score = sum(k in hay for k in keywords)
        if score > best_score:
            best, best_score = href, score
    if not best:
        return NodeResult("failed", error=f"no same-site link matches {keywords}")
    return NodeResult("done", output={"url": best})


# ---------------------------------------------------------------------------
# the approval gate and the only side effect
# ---------------------------------------------------------------------------


async def _gate(ex: Any) -> NodeResult:
    proposal = ex.inputs.get("proposal") or {}
    key = ex.node.idempotency_key
    if not key:
        return NodeResult("failed", error="approval_gate without an idempotency key")
    op = (
        "INSERT INTO approvals (idempotency_key, target_id, action_kind, proposed_action, reasoning, "
        "confidence, status, created_run_id, created_node_id) "
        "VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?) ON CONFLICT (idempotency_key) DO NOTHING",
        (key, ex.target["id"], proposal.get("action_kind", "unknown"),
         json.dumps(proposal, sort_keys=True), proposal.get("reasoning", ""),
         proposal.get("confidence"), ex.run_id, ex.node.node_id),
    )
    return NodeResult("blocked", extra_ops=[op], notes=[f"approval {key} queued"])


async def _execute(ex: Any) -> NodeResult:
    """Claim, then act, then record (review M10).

    1. CLAIM, one transaction: flip the row approved -> executed with
       result {"claimed_by": this node}. From here a human can no longer
       reject it, and nobody else can claim it. A row in any other state is
       refused: readiness should make that unreachable; this is the third
       independent check (D-019).
    2. ACT: the executor is idempotent by key (outbox file named by it).
    3. RECORD: the final result, in the same transaction as the node's done.
    A crash after 1 resumes here, sees its own claim, and finishes; a crash
    after 2 finds the file and does not send twice.
    """
    approval = ex.inputs.get("approval") or {}
    key = approval.get("idempotency_key")
    me = ex.node.node_id
    with ex.store._tx():
        row = ex.store.conn.execute(
            "SELECT status, action_kind, proposed_action, result FROM approvals WHERE idempotency_key = ?",
            (key,)).fetchone() if key else None
        if row is None:
            return NodeResult("failed", error=f"approval {key!r} is missing, not approved")
        claim = json.loads(row["result"] or "{}") if row["status"] == "executed" else {}
        if row["status"] == "approved":
            ex.store.conn.execute(
                "UPDATE approvals SET status = 'executed', "
                "executed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), executed_run_id = ?, result = ? "
                "WHERE idempotency_key = ? AND status = 'approved'",
                (ex.run_id, json.dumps({"claimed_by": me, "state": "claimed"}), key))
        elif not (row["status"] == "executed" and claim.get("claimed_by") == me):
            return NodeResult("failed", error=f"approval {key!r} is {row['status']}, not approved")
    payload = json.loads(row["proposed_action"])
    try:
        result = ex.services.actions.execute(row["action_kind"], payload, key)
    except Exception as exc:  # noqa: BLE001
        return NodeResult("failed", error=f"action failed after claim: {exc!r}")
    op = (
        "UPDATE approvals SET result = ? WHERE idempotency_key = ? AND status = 'executed'",
        (json.dumps({"claimed_by": me, "state": "done", **result}, sort_keys=True), key),
    )
    return NodeResult("done", output={"result": result}, extra_ops=[op])
