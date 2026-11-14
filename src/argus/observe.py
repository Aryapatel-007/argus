"""Read-only observability over the database: runs, traces, routing, health.

Every run is a trace and every node a span (ARGUS_BUILD 7.6). The spans are
reconstructed from `graph_nodes` + `node_events` + `llm_calls`; nothing is
exported anywhere (hard constraint 3). "Why did it pick this model?" is
answered from `routing_decisions.candidates` alone.

Shared by the CLI (`argus runs/trace/why/health`) and the dashboard.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from argus.storage import db


def _ts(s: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def _dur(a: Optional[str], b: Optional[str]) -> Optional[float]:
    ta, tb = _ts(a), _ts(b)
    return round((tb - ta).total_seconds(), 3) if ta and tb else None


def runs(db_path: Path, limit: int = 50) -> List[Dict[str, Any]]:
    with db.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT r.run_id, r.target_id, r.status, r.started_at, r.finished_at, r.error, "
            "g.graph_id, g.kind, g.status AS graph_status, g.tokens_used, g.seconds_used, "
            "(SELECT COUNT(*) FROM graph_nodes n WHERE n.graph_id = g.graph_id) AS nodes, "
            "(SELECT COUNT(*) FROM llm_calls c WHERE c.graph_id = g.graph_id) AS calls "
            "FROM runs r LEFT JOIN graphs g ON g.run_id = r.run_id "
            "WHERE r.target_id != 'profiler' ORDER BY r.started_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def model_sequence(conn, graph_id: str) -> List[str]:
    return [r["model_tag"] for r in conn.execute(
        "SELECT model_tag FROM llm_calls WHERE graph_id = ? AND role != 'embedder' ORDER BY created_at",
        (graph_id,))]


def swaps(seq: List[str]) -> int:
    """Model changes between consecutive GPU-lane calls."""
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b)


def trace(db_path: Path, run_id: str) -> Dict[str, Any]:
    with db.connect(db_path) as conn:
        graphs = [dict(r) for r in conn.execute(
            "SELECT * FROM graphs WHERE run_id = ? ORDER BY created_at", (run_id,))]
        out = {"run_id": run_id, "graphs": []}
        for g in graphs:
            nodes = [dict(r) for r in conn.execute(
                "SELECT n.*, d.model_tag, d.strategy, d.reason, d.decision_id FROM graph_nodes n "
                "LEFT JOIN routing_decisions d ON d.node_id = n.node_id AND d.superseded = 0 "
                "WHERE n.graph_id = ? ORDER BY n.created_at, n.node_id", (g["graph_id"],))]
            edges = [dict(r) for r in conn.execute(
                "SELECT src, dst, channel FROM graph_edges WHERE graph_id = ?", (g["graph_id"],))]
            calls = {}
            for c in conn.execute(
                    "SELECT node_id, COUNT(*) AS n, SUM(eval_count) AS tokens, AVG(tok_s) AS tok_s, "
                    "SUM(timed_out) AS timeouts, SUM(1 - ok) AS failed FROM llm_calls "
                    "WHERE graph_id = ? GROUP BY node_id", (g["graph_id"],)):
                calls[c["node_id"]] = dict(c)
            spans = []
            for n in nodes:
                spans.append({
                    "node_id": n["node_id"], "type": n["node_type"], "status": n["status"],
                    "attempt": n["attempt"], "retry_of": n["retry_of"], "model": n["model_tag"],
                    "strategy": n["strategy"], "reason": n["reason"], "error": n["error"],
                    "skip_reason": n["skip_reason"], "started_at": n["started_at"],
                    "finished_at": n["finished_at"], "duration_s": _dur(n["started_at"], n["finished_at"]),
                    "calls": calls.get(n["node_id"], {}),
                    "parents": sorted({e["src"] for e in edges if e["dst"] == n["node_id"]}),
                })
            seq = model_sequence(conn, g["graph_id"])
            out["graphs"].append({**g, "spans": spans, "swaps": swaps(seq), "model_sequence": seq,
                                  "edges": edges})
    return out


def why(db_path: Path, node_id: str) -> Dict[str, Any]:
    """Every routing decision for a node, with every candidate's breakdown."""
    with db.connect(db_path) as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM routing_decisions WHERE node_id = ? ORDER BY created_at", (node_id,))]
    for r in rows:
        r["candidates"] = json.loads(r["candidates"])
        r["models"] = json.loads(r["models"])
    return {"node_id": node_id, "decisions": rows}


def health(db_path: Path) -> List[Dict[str, Any]]:
    with db.connect(db_path) as conn:
        models = {r["model_tag"] for r in conn.execute("SELECT DISTINCT model_tag FROM model_profiles")}
        out = []
        for m in sorted(models):
            h = conn.execute("SELECT * FROM model_health WHERE model_tag = ?", (m,)).fetchone()
            recent = conn.execute(
                "SELECT AVG(tok_s) AS tok_s, SUM(timed_out) AS timeouts, COUNT(*) AS calls FROM "
                "(SELECT tok_s, timed_out FROM llm_calls WHERE model_tag = ? ORDER BY created_at DESC LIMIT 20)",
                (m,)).fetchone()
            prof = {r["metric"]: r["value"] for r in conn.execute(
                "SELECT metric, value FROM model_profiles WHERE model_tag = ? AND node_type = '*'", (m,))}
            arms = [dict(r) for r in conn.execute(
                "SELECT node_type, alpha, beta, pulls FROM arm_stats WHERE model_tag = ? ORDER BY node_type", (m,))]
            out.append({"model": m, "status": h["status"] if h else "healthy", "reason": h["reason"] if h else None,
                        "recent_tok_s": recent["tok_s"], "recent_calls": recent["calls"],
                        "recent_timeouts": recent["timeouts"], "profile": prof, "arms": arms})
    return out


def digest_of(db_path: Path, run_id: str) -> Optional[str]:
    with db.connect(db_path) as conn:
        r = conn.execute(
            "SELECT n.output FROM graph_nodes n JOIN graphs g ON g.graph_id = n.graph_id "
            "WHERE g.run_id = ? AND n.node_type = 'summarize' AND n.status = 'done'", (run_id,)).fetchone()
    return json.loads(r["output"])["digest"]["digest"] if r else None
