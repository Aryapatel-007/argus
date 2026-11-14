"""In-memory graph shapes shared by the validator, the store and templates.

These are plain data. Persistence lives in `store.py`; rules live in
`validator.py`. Keeping the shapes free of SQL lets the validator be a pure
function that hypothesis can hammer without a database.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional

NODE_STATUSES = ("pending", "ready", "running", "done", "failed", "blocked", "skipped")
TERMINAL_NODE_STATUSES = frozenset({"done", "failed", "skipped"})


@dataclass(frozen=True)
class Limits:
    """Hard caps. A planner can never grow a graph past these."""

    max_nodes: int = 120
    max_depth: int = 24
    max_expansions: int = 8
    max_attempts: int = 3

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Limits":
        data = data or {}
        return cls(**{k: int(v) for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, int]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class NodeSpec:
    node_id: str
    node_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    attempt: int = 1
    retry_of: Optional[str] = None
    #: The original node of a retry chain. Equal to node_id for attempt 1.
    retry_root: Optional[str] = None
    #: Required (and unique) on side-effect nodes. Generated at creation and
    #: stored with the node, i.e. BEFORE the node can run.
    idempotency_key: Optional[str] = None
    #: None means "use the node type's default".
    prunable: Optional[bool] = None
    #: Planner's difficulty signal in [0, 1]; the router reads it.
    difficulty: float = 0.5
    status: str = "pending"
    #: Which template instance this node belongs to, for procedural memory.
    template_instance: Optional[str] = None

    def __post_init__(self) -> None:
        if self.retry_root is None:
            self.retry_root = self.node_id

    def clone(self, **changes: Any) -> "NodeSpec":
        return replace(self, **changes)


@dataclass(frozen=True)
class EdgeSpec:
    src: str
    dst: str
    channel: str


@dataclass
class GraphSpec:
    nodes: Dict[str, NodeSpec] = field(default_factory=dict)
    edges: List[EdgeSpec] = field(default_factory=list)

    @classmethod
    def build(cls, nodes: Iterable[NodeSpec], edges: Iterable[EdgeSpec]) -> "GraphSpec":
        return cls({n.node_id: n for n in nodes}, list(edges))

    def copy(self) -> "GraphSpec":
        return GraphSpec(
            {k: v.clone(params=dict(v.params)) for k, v in self.nodes.items()},
            list(self.edges),
        )

    def merged(self, other: "GraphSpec") -> "GraphSpec":
        merged = self.copy()
        for node_id, node in other.nodes.items():
            merged.nodes[node_id] = node
        merged.edges.extend(other.edges)
        return merged

    def parents(self, node_id: str) -> List[EdgeSpec]:
        return [e for e in self.edges if e.dst == node_id]

    def children(self, node_id: str) -> List[EdgeSpec]:
        return [e for e in self.edges if e.src == node_id]

    def roots(self) -> List[str]:
        has_parent = {e.dst for e in self.edges}
        return [n for n in self.nodes if n not in has_parent]


def new_node_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def idempotency_key_for(graph_id: str, node_id: str) -> str:
    """Deterministic per node, unique per graph: sha256(graph_id, node_id)."""
    digest = hashlib.sha256(f"{graph_id}\x00{node_id}".encode("utf-8")).hexdigest()
    return f"idem-{digest[:32]}"
