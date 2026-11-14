"""Subgraph templates: procedural memory for the planner.

The planner never invents raw graph structure. It picks templates by name and
fills their params; `instantiate` turns that into nodes and edges with fresh
ids. Small local models are reliable at "pick from this list and fill these
fields" and unreliable at "emit a DAG", so this is where structured output
stays tractable.

Built-in templates live here. Learned templates (a replan-produced subgraph
that succeeded) are stored in the `templates` table as a node-type sequence
and become usable once promoted; see `store.record_template_outcome`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

from argus.graph.model import EdgeSpec, GraphSpec, NodeSpec, idempotency_key_for, new_node_id

#: (subgraph, exports). exports names the nodes other wiring attaches to,
#: e.g. {"changes": <diff node id>}.
Built = Tuple[GraphSpec, Dict[str, str]]


@dataclass(frozen=True)
class Template:
    name: str
    description: str
    required: Tuple[str, ...]
    build: Callable[[str, str, Dict[str, Any], str], Built]
    planner_visible: bool = True
    defaults: Dict[str, Any] = field(default_factory=dict)

    def missing_params(self, params: Dict[str, Any]) -> List[str]:
        return [p for p in self.required if params.get(p) in (None, "", [])]


def _n(node_type: str, instance: str, **kw: Any) -> NodeSpec:
    return NodeSpec(node_id=new_node_id(node_type), node_type=node_type,
                    template_instance=instance, **kw)


def build_check_page(graph_id: str, target_id: str, params: Dict[str, Any], instance: str) -> Built:
    """fetch -> extract -> critique -> resolve -> diff -> write_fact -> remember.

    recall runs in parallel with the fetch (I/O lane), which is exactly the
    overlap the linear loop could not do.
    """
    fields = list(params.get("fields") or [])
    fetch = _n("fetch_page", instance, params={"url": params["url"]})
    extract = _n("extract", instance, params={"fields": fields, "target_id": target_id},
                 difficulty=float(params.get("difficulty", 0.5)))
    # The extract's trigger (fields missing -> replan) must name the fetch
    # node holding the page, so a follow_link template can search its links.
    extract.params["_fetch_node"] = fetch.node_id
    critique = _n("critique", instance, params={"fields": fields})
    recall = _n("recall", instance, params={"target_id": target_id})
    resolve = _n("resolve_entity", instance, params={"target_id": target_id, "fields": fields})
    diff = _n("diff", instance, params={"target_id": target_id, "fields": fields,
                                        "source_url": params["url"]})
    write = _n("write_fact", instance, params={"target_id": target_id})
    remember = _n("remember", instance, params={"target_id": target_id, "source_url": params["url"]})
    nodes = [fetch, extract, critique, recall, resolve, diff, write, remember]
    edges = [
        EdgeSpec(fetch.node_id, extract.node_id, "page"),
        EdgeSpec(extract.node_id, critique.node_id, "subject"),
        EdgeSpec(fetch.node_id, critique.node_id, "evidence"),
        EdgeSpec(critique.node_id, resolve.node_id, "facts"),
        EdgeSpec(recall.node_id, resolve.node_id, "memory"),
        EdgeSpec(resolve.node_id, diff.node_id, "resolved"),
        EdgeSpec(recall.node_id, diff.node_id, "memory"),
        EdgeSpec(fetch.node_id, diff.node_id, "evidence"),
        EdgeSpec(diff.node_id, write.node_id, "changes"),
        EdgeSpec(write.node_id, remember.node_id, "durable"),
    ]
    return GraphSpec.build(nodes, edges), {"changes": diff.node_id, "page": fetch.node_id}


def build_follow_link(graph_id: str, target_id: str, params: Dict[str, Any], instance: str) -> Built:
    """search a fetched page for a link, then run check_page on that link.

    Used by replan when a page says "see the fees page" and the fields asked
    for are not on the first page. `from_node` is the fetch_page that holds
    the links.
    """
    search = _n("search", instance, params={"keywords": list(params.get("keywords") or [])})
    sub, exports = build_check_page(graph_id, target_id,
                                    {"url": params.get("fallback_url", ""), "fields": params["fields"]},
                                    instance)
    fetch_id = exports["page"]
    sub.nodes[fetch_id].params["url"] = None  # comes from the search
    sub.nodes[search.node_id] = search
    sub.edges.append(EdgeSpec(params["from_node"], search.node_id, "page"))
    sub.edges.append(EdgeSpec(search.node_id, fetch_id, "url"))
    return sub, exports


def build_digest(graph_id: str, target_id: str, params: Dict[str, Any], instance: str) -> Built:
    """summarize over every page's changes. Not planner-visible: the engine adds it."""
    summarize = _n("summarize", instance, params={"target_id": target_id})
    sub = GraphSpec.build([summarize], [
        EdgeSpec(src, summarize.node_id, "changes") for src in params.get("changes_from", [])
    ])
    return sub, {"digest": summarize.node_id}


def build_notify(graph_id: str, target_id: str, params: Dict[str, Any], instance: str) -> Built:
    """propose -> approval_gate -> execute. Not planner-visible.

    Idempotency keys are generated HERE, at creation, and stored with the
    nodes when the expansion is stored, i.e. before either node can run.
    """
    propose = _n("propose_action", instance, params={"target_id": target_id})
    gate = _n("approval_gate", instance, params={"target_id": target_id})
    execute = _n("execute_action", instance, params={"target_id": target_id})
    gate.idempotency_key = idempotency_key_for(graph_id, gate.node_id)
    execute.idempotency_key = idempotency_key_for(graph_id, execute.node_id)
    gate.prunable = False
    execute.prunable = False
    sub = GraphSpec.build([propose, gate, execute], [
        EdgeSpec(params["digest_from"], propose.node_id, "digest"),
        EdgeSpec(propose.node_id, gate.node_id, "proposal"),
        EdgeSpec(gate.node_id, execute.node_id, "approval"),
    ])
    return sub, {"gate": gate.node_id, "action": execute.node_id}


def build_reflect(graph_id: str, target_id: str, params: Dict[str, Any], instance: str) -> Built:
    recall = _n("recall", instance, params={"target_id": target_id, "all_current": True})
    reflect = _n("reflect", instance, params={"target_id": target_id})
    remember = _n("remember", instance, params={"target_id": target_id, "source_url": None,
                                                "summary": True})
    return GraphSpec.build([recall, reflect, remember], [
        EdgeSpec(recall.node_id, reflect.node_id, "memory"),
        EdgeSpec(reflect.node_id, remember.node_id, "durable"),
    ]), {}


TEMPLATES: Dict[str, Template] = {
    t.name: t
    for t in [
        Template("check_page", "Fetch one page, extract the requested fields, review, "
                 "compare with memory, and store what changed.",
                 ("url", "fields"), build_check_page),
        Template("follow_link", "Find a link on an already fetched page by keywords and "
                 "check that linked page for the requested fields.",
                 ("from_node", "keywords", "fields"), build_follow_link),
        Template("digest", "Summarize every page's changes.", ("changes_from",), build_digest,
                 planner_visible=False),
        Template("notify", "Draft a message, wait for approval, then send it.",
                 ("digest_from",), build_notify, planner_visible=False),
        Template("reflect", "Consolidate a target's facts into higher-level summaries.",
                 (), build_reflect, planner_visible=False),
    ]
}


def planner_catalog() -> str:
    lines = []
    for t in TEMPLATES.values():
        if t.planner_visible:
            lines.append(f"- `{t.name}` (params: {', '.join(t.required)}): {t.description}")
    return "\n".join(lines)


def instantiate(name: str, graph_id: str, target_id: str, params: Dict[str, Any]) -> Built:
    template = TEMPLATES[name]
    missing = template.missing_params(params)
    if missing:
        raise ValueError(f"template {name!r} missing params {missing}")
    instance = new_node_id(f"tpl-{name}")
    return template.build(graph_id, target_id, {**template.defaults, **params}, instance)
