# Graph engine spec (Phase E)

Status: design, 2026-11-14. Implements ARGUS_BUILD.md section 6.
Package: `src/argus/graph/`. Decisions referenced as D-NNN live in
`notes/decisions-log.md`.

## 1. Constraints that shape everything

- One chat model in VRAM at a time; a swap costs a cold load (~6s for
  qwen3.5:9b). Swap cost is a term in every routing decision and the GPU
  lane drains the resident model's queue before swapping.
- `think=False` top-level on every chat call, every model. 120s hard timeout.
  Every call (including timeouts) logs one `llm_calls` row with tok/s.
- Every write through `db.atomic()`. No transaction is open while a model,
  MCP or browser call is awaited: compute, then write.
- A crash at any instant resumes with no duplicated side effect and no
  regenerated stored output.
- Nothing external without an approved `approvals` row, enforced by graph
  structure (validator) and again at execution time (status check).

## 2. Data model (migrations 003, 004, 005)

One migration per concern (D-010):

- **003_graph.sql** — `graphs`, `graph_nodes`, `graph_edges`,
  `graph_expansions`, `node_events`, `node_samples`, `routing_decisions`,
  `templates`.
- **004_models.sql** — `model_profiles`, `co_residency`, `arm_stats`,
  `arm_rewards`, `model_health`.
- **005_llm_calls_graph.sql** — rebuild `llm_calls` (role CHECK widened to
  node types, plus `graph_id`, `node_id`, `strategy`, `sample_index`); add
  `approvals.created_node_id`, `approvals.decided_by`, `approvals.decision_note`.

Node status: `pending -> ready -> running -> done | failed`, plus
`blocked` (approval gate waiting on a human) and `skipped` (unreachable,
rejected gate, or pruned by budget). Terminal: done, failed, skipped.

`node_events` is the checkpoint log: one row per status change, written in
the same transaction as the `graph_nodes` update. `graph_nodes` holds current
state; resume reads it directly.

## 3. Node types (`nodes.py`)

Each `NodeType` declares: `name`, `kind` (llm | tool | pure | gate | action),
`input_names`, `output_model` (pydantic), `side_effect` (bool),
`validator(output, inputs) -> list[str]`, `default_strategy`,
`prunable` (bool), `prompt_file`.

| type | kind | side effect | notes |
|---|---|---|---|
| plan | llm | no | output is an expansion (templates + params) |
| replan | llm | no | output is an expansion attached to an existing node |
| search | tool | no | link search inside an already fetched page (D-011) |
| fetch_page | tool | no | browser MCP server, I/O lane |
| extract | llm | no | facts from page text; grounding validator |
| resolve_entity | pure+llm | no | normalized key, embedding sim, model check |
| recall | tool | no | memory MCP `recall_facts` / current facts |
| diff | pure | no | added / modified / unchanged / missing |
| critique | llm | no | verdict; never the subject's model |
| write_fact | llm | no | durable phrasing + importance |
| remember | tool | no (local) | memory MCP `remember_fact`, idempotent by hash |
| summarize | llm | no | digest of changes only; silence when none |
| propose_action | llm | no | cross_model required |
| approval_gate | gate | writes approvals row | blocks until a human decides |
| execute_action | action | YES | only after an approved gate |
| reflect | llm | no | cluster facts, write summary facts |

## 4. Edges, readiness, retries

Edge = (src, dst, input_name). For each input name of a node, the
candidate sources are its parents on that name. The input is **satisfied**
when some source is `done` and no source with a higher `attempt` is still
non-terminal; the value is the output of the done source with the highest
attempt. A node is ready when every input is satisfied and, for
`execute_action`, its gate ancestors are `done` (approved).

Retries are new nodes, never back edges (D-012). A failed node X with
attempts left gets a clone X' (`attempt+1`, `retry_of = X`) with the same
parents, and edges X' -> each child of X on the same input names. A
critique that rejects its subject fails; the retry policy clones both the
subject and the critique, feeding the critique's reflection into the
subject clone's params (Reflexion). When retries are exhausted, every node
whose input can no longer be satisfied becomes `skipped`.

## 5. Validator (`validator.py`)

Pure function over (nodes, edges, limits); also run on (existing graph +
proposed expansion) before an expansion is stored. Violations block
execution of the whole graph or rejection of the expansion. Checks:

1. Acyclic (Kahn's algorithm).
2. Every `execute_action` has an `approval_gate` on **every** path from a
   root: removing all gates leaves no path root -> execute_action.
   Equivalently, execute_action is unreachable in the graph with gates
   deleted. Also every execute_action must have at least one gate ancestor.
3. Every side-effect node (`execute_action`, `approval_gate`) carries a
   non-empty idempotency key, unique within the graph.
4. Limits: nodes per graph, depth, expansions, attempts per node.
5. Every input name of every node has at least one parent, and each
   parent's node type lists that name in its `provides` set (schema
   satisfiability).
6. Gates and side-effect nodes are never `prunable`.

Property tests (hypothesis) generate random DAGs, random gate placements
and random expansions; a deliberately broken validator (gate check removed)
must let a bypass through (negative control).

## 6. Router (`router.py`)

For a ready LLM node, every allowed healthy chat model is a candidate.
Hard filters: critique/judge model != subject model; cross_model needs two
distinct models; degraded models excluded unless nothing else is left.

    quality   = Thompson sample from Beta(alpha, beta) (learning on)
                or posterior mean (learning off / exploration cap hit)
    latency_s = (0 if resident or co-resident else cold_load_s)
                + expected_tokens / tok_s
    swap_s    = 0 if resident/co-resident else cold_load_s
    risk      = 1 - json_valid_rate, raised by recent timeouts
    utility   = w_q*quality - w_l*latency_s/latency_scale
                - w_s*swap_s/latency_scale - w_r*risk

Weights, scales, prior strength and exploration cap live in
`config/routing.yaml`, never in source. Every candidate's components are
stored in `routing_decisions.candidates` (JSON), so "why this model" is a
SQL query. Thompson sampling chosen over UCB (D-013).

Determinism: a decision is stored when the node becomes ready. The GPU lane
may *revise* a not-yet-started decision to the resident model if the
utility loss is under `revise_max_loss`; the old row is marked superseded
and the new row records `revised_from`. When the node starts, the
decision is frozen in the same transaction as `ready -> running`. On
resume, a frozen decision is reused, never recomputed.

## 7. Strategies (`strategies.py`)

`single`, `self_consistency` (N samples, seeds derived from node id and
sample index, majority on a canonical key), `cascade` (cheapest adequate
first, escalate on validator reject), `cross_model` (two models, must
agree, required for propose_action), `reflexion_retry` (the retry
mechanism of section 4). Every completed sample is stored in
`node_samples` before the next sample starts; resume reuses stored samples.

## 8. Scheduler (`scheduler.py`)

asyncio. GPU lane: one worker, one call at a time. I/O lane: semaphore
(default 4). Loop: route newly ready nodes, dispatch tool/pure nodes to the
I/O lane, enqueue LLM nodes to the GPU lane. The GPU worker picks the next
node from the resident model's group first; otherwise the largest group.

Health: rolling tok/s per model vs its profiled tok/s. Ratio below
`degrade_ratio` on two consecutive calls, or two consecutive timeouts,
marks the model degraded (row in `model_health`). After
`health_cooldown_s` a one-line probe call restores it if tok/s recovers.

Budgets: per graph wall-clock seconds and tokens. Below
`prune_fraction` remaining, not-started `prunable` nodes become skipped
with reason `budget`. Gates and side-effect nodes are never prunable.

## 9. Crash-resume

| crash point | on disk | resume does |
|---|---|---|
| between node transitions | last committed node states | continue |
| LLM call in flight | node `running`, decision frozen, maybe llm_calls row | node -> ready, same decision, call again |
| between samples | completed samples in node_samples | reuse them, run the rest |
| expansion computed, not stored | replan node running | rerun replan (output never stored) |
| expansion stored | nodes in graph | continue |
| approval_gate running | maybe approvals row (UNIQUE key) | INSERT OR IGNORE, read back |
| execute_action running | approvals status approved, maybe external effect done | executor checks effect by idempotency key before acting |
| remember in flight | some facts written | re-call; memory server returns unchanged for written ones; change kinds come from `diff` output, not from remember |

Resume rule: every `running` node goes back to `ready` in one transaction
with a `node_events` row (`reason: resume`). Nothing else is recomputed.

## 10. Learning (`bandit.py`)

Arm = (node_type, model_tag). Beta posterior in `arm_stats`, seeded from
profiler quality with strength `prior_strength`. Rewards in `arm_rewards`
with UNIQUE(node_id, source): validator pass/fail, critic verdict (only when
the critic model differs from the subject model), eval harness verdict.
The UNIQUE key makes reward application idempotent across resume.

## 11. Templates (`templates.py`)

Built-in templates (e.g. `check_page`: fetch_page -> extract -> critique ->
recall -> resolve_entity -> diff -> write_fact -> remember). The planner
chooses templates by name and fills params; it never invents raw structure.
A run whose template instance finishes with all nodes done and critic pass
increments the template's success count; a replan-produced subgraph that
succeeds is stored as a `learned` template candidate.
