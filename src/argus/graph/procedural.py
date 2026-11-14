"""Procedural memory: which subgraph shapes work, per template and target.

After every finished graph, each template instance gets ONE outcome row
(`template_outcomes`, keyed by instance, so a resumed graph cannot count
twice). An instance succeeded if every retry chain in it ended with a done
node. Built-in templates accumulate uses/successes. A step that a REPLAN
chose and that succeeded becomes a `learned` template candidate for that
target; after `PROMOTE_AFTER` successes it is promoted and the planner is
shown it as a known-good step (`hints_for`), so the system learns how to
check a target, not only what it saw there.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List

from argus.storage import db

PROMOTE_AFTER = 2


def template_name(instance: str) -> str:
    # instance ids are "tpl-<name>-<hex>" (templates.instantiate)
    return instance[4:].rsplit("-", 1)[0] if instance.startswith("tpl-") else instance


def _succeeded(members: List[Any]) -> bool:
    """Every chain ended done, or was deliberately skipped (nothing to send,
    a human said no, budget). Only a chain that FAILED counts against the
    template: a quiet week is not a broken notify template (review M7)."""
    chains: Dict[str, List[Any]] = defaultdict(list)
    for n in members:
        chains[n.retry_root].append(n)
    return all(any(m.status in ("done", "skipped") for m in chain) and
               not all(m.status == "failed" for m in chain)
               for chain in chains.values())


def record_graph(store: Any, graph_id: str) -> None:
    nodes = store.nodes(graph_id)
    g = store.graph(graph_id)
    by_instance: Dict[str, List[Any]] = defaultdict(list)
    for n in nodes.values():
        if n.template_instance:
            by_instance[n.template_instance].append(n)
    replan_instances = set()
    for r in store.conn.execute(
            "SELECT spec FROM graph_expansions WHERE graph_id = ? AND kind = 'replan' AND accepted = 1",
            (graph_id,)):
        for nd in json.loads(r["spec"]).get("nodes", []):
            if nd.get("template_instance"):
                replan_instances.add(nd["template_instance"])
    with store._tx():  # lease-guarded like every other graph write
        for instance, members in by_instance.items():
            name = template_name(instance)
            ok = _succeeded(members)
            template_id = name
            pattern = {"template": name}
            origin = "builtin"
            if instance in replan_instances:
                origin = "learned"
                params = {}
                for m in members:
                    if m.node_type == "search":
                        params["keywords"] = m.params.get("keywords")
                    if m.node_type == "extract":
                        params["fields"] = m.params.get("fields")
                pattern = {"template": name, "target_id": g.target_id, "params": params}
                template_id = f"learned:{g.target_id}:{name}:{json.dumps(params, sort_keys=True)}"
            store.conn.execute(
                "INSERT INTO templates (template_id, name, origin, pattern) VALUES (?, ?, ?, ?) "
                "ON CONFLICT (template_id) DO NOTHING",
                (template_id, name, origin, json.dumps(pattern, sort_keys=True)))
            cur = store.conn.execute(
                "INSERT INTO template_outcomes (template_instance, template_id, graph_id, success) "
                "VALUES (?, ?, ?, ?) ON CONFLICT (template_instance) DO NOTHING",
                (instance, template_id, graph_id, int(ok)))
            if cur.rowcount == 1:
                store.conn.execute(
                    "UPDATE templates SET uses = uses + 1, successes = successes + ?, "
                    "promoted = CASE WHEN origin = 'learned' AND successes + ? >= ? THEN 1 ELSE promoted END, "
                    "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE template_id = ?",
                    (int(ok), int(ok), PROMOTE_AFTER, template_id))


def hints_for(conn: Any, target_id: str) -> List[Dict[str, Any]]:
    """Promoted learned steps for this target, best first."""
    rows = conn.execute(
        "SELECT pattern, uses, successes FROM templates WHERE origin = 'learned' AND promoted = 1 "
        "ORDER BY successes DESC, uses ASC").fetchall()
    out = []
    for r in rows:
        p = json.loads(r["pattern"])
        if p.get("target_id") == target_id:
            out.append({**p, "successes": r["successes"], "uses": r["uses"]})
    return out
