"""Node type registry: every unit of graph work is one of these types.

A NodeType declares what it consumes (input channels), what it feeds
(`provides`), whether it calls a model, a tool, or nothing, whether it has an
external side effect, its deterministic validator, and its default execution
strategy. The validator in `validator.py` uses `inputs`/`provides` to prove
every node's inputs are satisfiable by its parents before anything runs.

Output models are pydantic so the same class produces the JSON schema passed
to Ollama's `format=` and validates what comes back.

Deterministic validators run BEFORE any critic. They return a list of
human-readable problems; empty means the output passed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type

from pydantic import BaseModel, Field

Validator = Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], List[str]]
"""(output, inputs, params) -> problems."""

KINDS = ("llm", "tool", "pure", "gate", "action")


def _no_checks(output: Dict[str, Any], inputs: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
    return []


@dataclass(frozen=True)
class NodeType:
    name: str
    kind: str
    #: Input channels that must be satisfied before the node is ready.
    inputs: Tuple[str, ...] = ()
    #: Inputs that collect EVERY parent chain (a list), instead of exactly one.
    fan_in: Tuple[str, ...] = ()
    #: Channels (from inputs or fan_in) allowed to have no parent at all.
    optional_inputs: Tuple[str, ...] = ()
    #: Channels this node's output can feed.
    provides: Tuple[str, ...] = ()
    output_model: Optional[Type[BaseModel]] = None
    side_effect: bool = False
    #: Budget pruning may skip a not-yet-started node of this type.
    prunable: bool = True
    default_strategy: str = "single"
    prompt_file: Optional[str] = None
    validator: Validator = field(default=_no_checks)
    #: llm_calls.role value for calls made by this node type.
    role: Optional[str] = None


# ---------------------------------------------------------------------------
# output models
# ---------------------------------------------------------------------------


class PlanStep(BaseModel):
    template: str = Field(description="Name of a subgraph template from the catalog.")
    params: Dict[str, Any] = Field(default_factory=dict)


class PlanOutput(BaseModel):
    thought: str
    steps: List[PlanStep] = Field(default_factory=list)


class ExtractedFact(BaseModel):
    entity_key: str = Field(description="Which requested field this fact is about.")
    content: str = Field(description="The fact, one sentence, quoted or close to the page.")
    attributes: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractOutput(BaseModel):
    facts: List[ExtractedFact] = Field(default_factory=list)


class CritiqueOutput(BaseModel):
    verdict: Literal["pass", "fail"]
    confidence: float = Field(ge=0.0, le=1.0)
    reflection: str


class DurableFact(BaseModel):
    entity_key: str
    content: str
    importance: float = Field(ge=0.0, le=1.0)


class WriteFactOutput(BaseModel):
    facts: List[DurableFact] = Field(default_factory=list)


class SummarizeOutput(BaseModel):
    digest: str


class ProposeActionOutput(BaseModel):
    action_kind: Literal["outbox_message"]
    subject: str
    body: str
    reasoning: str


class EntityMatch(BaseModel):
    same_entity: bool
    confidence: float = Field(ge=0.0, le=1.0)


class ReflectSummary(BaseModel):
    theme: str
    content: str
    importance: float = Field(ge=0.0, le=1.0)


class ReflectOutput(BaseModel):
    summaries: List[ReflectSummary] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# deterministic validators
# ---------------------------------------------------------------------------

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numeric_tokens(text: str) -> List[str]:
    """Numbers as written, commas stripped: '45,000' -> '45000'."""
    return [m.group(0).replace(",", "") for m in _NUMBER.finditer(text or "")]


def ungrounded_numbers(claim: str, evidence: str) -> List[str]:
    """Numbers in `claim` that never appear in `evidence`.

    The cheap faithfulness check that runs before any critic: a model that
    writes a fee or a date the page never said is caught without a second
    model call. Measured motivation: notes/writer-probe-results.md set C.
    """
    seen = set(numeric_tokens(evidence))
    return [n for n in numeric_tokens(claim) if n not in seen]


def normalize_key(key: str) -> str:
    """Canonical entity_key form: lowercase, non-alphanumerics to '_'."""
    return re.sub(r"[^a-z0-9]+", "_", (key or "").lower()).strip("_")


def _validate_plan(output, inputs, params) -> List[str]:
    from argus.graph import templates  # local: templates imports nodes

    problems = []
    if not output.get("steps"):
        problems.append("plan has no steps")
    for i, step in enumerate(output.get("steps", [])):
        name = step.get("template")
        if name not in templates.TEMPLATES:
            problems.append(f"step {i}: unknown template {name!r}; known: {sorted(templates.TEMPLATES)}")
            continue
        missing = templates.TEMPLATES[name].missing_params(step.get("params") or {})
        if missing:
            problems.append(f"step {i}: template {name!r} missing params {missing}")
    return problems


def _validate_extract(output, inputs, params) -> List[str]:
    problems = []
    page = inputs.get("page") or {}
    text = page.get("text") or ""
    wanted = {normalize_key(f) for f in params.get("fields", [])}
    for i, fact in enumerate(output.get("facts", [])):
        content = (fact.get("content") or "").strip()
        if not content:
            problems.append(f"fact {i}: empty content")
            continue
        key = normalize_key(fact.get("entity_key", ""))
        if wanted and key not in wanted:
            problems.append(f"fact {i}: entity_key {fact.get('entity_key')!r} not in requested fields {sorted(wanted)}")
        bad = ungrounded_numbers(content, text)
        if bad:
            problems.append(f"fact {i}: numbers {bad} do not appear on the page")
    return problems


def _validate_write_fact(output, inputs, params) -> List[str]:
    problems = []
    changes = inputs.get("changes") or {}
    evidence = " ".join(
        str(c.get("content", "")) for c in changes.get("items", [])
    )
    for i, fact in enumerate(output.get("facts", [])):
        if not (fact.get("content") or "").strip():
            problems.append(f"fact {i}: empty content")
        bad = ungrounded_numbers(fact.get("content", ""), evidence)
        if bad:
            problems.append(f"fact {i}: numbers {bad} not in the extracted facts")
    return problems


def _validate_summarize(output, inputs, params) -> List[str]:
    digest = output.get("digest") or ""
    changes = inputs.get("changes") or []
    evidence_parts = []
    for block in changes:
        for item in (block or {}).get("items", []):
            evidence_parts.append(str(item.get("content", "")))
            evidence_parts.append(str(item.get("previous_content") or ""))
    bad = ungrounded_numbers(digest, " ".join(evidence_parts))
    return [f"digest numbers {bad} appear in no input fact"] if bad else []


def _validate_propose(output, inputs, params) -> List[str]:
    problems = []
    if not (output.get("body") or "").strip():
        problems.append("empty body")
    digest = (inputs.get("digest") or {}).get("digest", "")
    bad = ungrounded_numbers(output.get("body", ""), digest)
    if bad:
        problems.append(f"body numbers {bad} not in the digest")
    return problems


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

NODE_TYPES: Dict[str, NodeType] = {}


def _register(nt: NodeType) -> NodeType:
    if nt.kind not in KINDS:
        raise ValueError(f"bad kind {nt.kind!r}")
    NODE_TYPES[nt.name] = nt
    return nt


_register(NodeType("plan", "llm", provides=(), output_model=PlanOutput,
                   prunable=False, prompt_file="plan.md", validator=_validate_plan, role="plan"))
_register(NodeType("replan", "llm", inputs=("trigger",), provides=(), output_model=PlanOutput,
                   prompt_file="plan.md", validator=_validate_plan, role="replan"))
_register(NodeType("search", "tool", inputs=("page",), provides=("url",)))
_register(NodeType("fetch_page", "tool", fan_in=("url",), optional_inputs=("url",),
                   provides=("page", "evidence", "trigger"),
                   prunable=False))
_register(NodeType("extract", "llm", inputs=("page",), provides=("subject", "trigger"),
                   output_model=ExtractOutput, prompt_file="extract.md",
                   validator=_validate_extract, role="extract", prunable=False))
# critique passes its subject through on the `facts` channel only when the
# verdict is pass, so nothing downstream of extract can run on facts the
# critic has not approved. A rejected extract is retried as a new node pair
# (extract', critique') instead of leaking into resolve/diff.
_register(NodeType("critique", "llm", inputs=("subject", "evidence"), provides=("verdict", "facts"),
                   output_model=CritiqueOutput, prompt_file="critique.md", role="critique",
                   # Not prunable: write_fact waits on a PASS verdict. Pruning the
                   # critic would mean either skipping the write or saving an
                   # unreviewed fact; neither is acceptable to save time.
                   prunable=False))
_register(NodeType("recall", "tool", provides=("memory",), prunable=False))
# resolve_entity is routed like an LLM node: most facts resolve by key with no
# model call, but ambiguous ones need a model check (entity_match prompt), and
# that call must go through the GPU lane with a stored routing decision.
_register(NodeType("resolve_entity", "llm", inputs=("facts", "memory"), provides=("resolved",),
                   role="resolve_entity", prunable=False))
# diff reads the page too: a stored fact whose words are still on the page is
# unchanged, however the extractor phrased it this time (D-016).
_register(NodeType("diff", "pure", inputs=("resolved", "memory", "evidence"), provides=("changes",),
                   prunable=False))
_register(NodeType("write_fact", "llm", inputs=("changes",),
                   provides=("durable",), output_model=WriteFactOutput,
                   prompt_file="write_fact.md", validator=_validate_write_fact,
                   role="write_fact", prunable=False))
_register(NodeType("remember", "tool", inputs=("durable",), provides=("stored",), prunable=False))
_register(NodeType("summarize", "llm", fan_in=("changes",), provides=("digest",),
                   output_model=SummarizeOutput, prompt_file="summarize.md",
                   validator=_validate_summarize, role="summarize", prunable=False))
_register(NodeType("propose_action", "llm", inputs=("digest",), provides=("proposal",),
                   output_model=ProposeActionOutput, prompt_file="propose_action.md",
                   validator=_validate_propose, default_strategy="cross_model",
                   role="propose_action"))
_register(NodeType("approval_gate", "gate", inputs=("proposal",), provides=("approval",),
                   side_effect=True, prunable=False))
_register(NodeType("execute_action", "action", inputs=("approval",), provides=("result",),
                   side_effect=True, prunable=False))
_register(NodeType("reflect", "llm", inputs=("memory",), provides=("durable",),
                   output_model=ReflectOutput, prompt_file="reflect.md", role="reflect"))

#: Types whose execution has an effect outside this graph's own tables.
SIDE_EFFECT_TYPES = frozenset(n for n, t in NODE_TYPES.items() if t.side_effect)

#: Every llm_calls.role a graph node may log (migration 005 CHECK mirrors it).
GRAPH_ROLES = tuple(sorted({t.role for t in NODE_TYPES.values() if t.role}))


def get(name: str) -> NodeType:
    try:
        return NODE_TYPES[name]
    except KeyError:
        raise KeyError(f"unknown node type {name!r}; known: {sorted(NODE_TYPES)}") from None


def accepts(node_type: str, channel: str) -> bool:
    nt = get(node_type)
    return channel in nt.inputs or channel in nt.fan_in


def schema_for(node_type: str) -> Dict[str, Any]:
    model = get(node_type).output_model
    if model is None:
        raise ValueError(f"{node_type} has no output model")
    return model.model_json_schema()
