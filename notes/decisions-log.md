# Decisions log

Numbered. Each entry: what was chosen, what was rejected, and why.
Sprint 0-2 decisions live in `design-decisions.md` at the repo root and still
stand unless an entry here explicitly supersedes one.

---

## D-001 — Design decisions are made in this session, not an architect chat

**Chose:** the build session decides design questions itself and records
each one here.
**Rejected:** the old CLAUDE.md rule that every design question waits for a
separate architect chat.
**Why:** ARGUS_BUILD.md (the end-to-end build directive) replaces sprints
with self-run phases. Blocking on a separate chat for every choice would stall
a build that is meant to resume from disk across sessions. The hard
constraints and the stop conditions in ARGUS_BUILD.md section 1.4 still
override anything decided here.

## D-002 — Raw Playwright plus our own extraction, not Browser Use

**Chose:** the browser MCP server drives Playwright directly (Chromium
headless). Page understanding is an `extract` graph node that runs a local
model over the page text.
**Rejected:** browser-use 1.0.4.
**Why:** inspected the wheel before installing (notes/environment.md).
Telemetry is on by default, it calls PyPI for update checks at runtime, it
has code paths aimed at its hosted API, it has no native Ollama provider, and
it pins `mcp` and `pydantic` to exact versions different from ours. Hard
constraint 3 (no telemetry) rules out a default-on phone-home. The original
reason for Browser Use (selectors are brittle across uncontrolled sites) is
still answered: we never write per-site selectors. We fetch rendered text
and let the extract node read it. This supersedes the "Browser Use layered on
Playwright" entry in design-decisions.md.

## D-003 — Dashboard: FastAPI + Jinja2, server-rendered

**Chose:** FastAPI app with Jinja2 templates and plain HTML/CSS, no JS build
step, served by uvicorn on 127.0.0.1 only.
**Rejected:** Streamlit.
**Why:** Streamlit collects usage stats by default (hard constraint 3), and
its rerun-the-script model fights a dashboard whose job is mostly reading
SQLite and posting approve/reject. FastAPI is also already half-present
(starlette and uvicorn arrive with mcp).

## D-004 — In-repo eval harness, not deepeval

**Chose:** build the evaluation harness in `src/argus/eval/`, judged by a
local model that is never the model being judged.
**Rejected:** deepeval 4.3.4.
**Why:** hard dependency on posthog and confident-trace (telemetry, opt-out
not opt-in), plus a `click<8.4.0` pin that conflicts with the installed
environment. Its metrics also assume a hosted judge by default. Our judge
needs a "different model from the one judged" rule and a calibration set
anyway, which deepeval would not give us for free.

## D-005 — pytest as the runner, existing unittest suites kept as-is

**Chose:** `python -m pytest` with `pytest.ini` setting `pythonpath = src`.
New tests use plain pytest functions and hypothesis.
**Rejected:** rewriting the 179 existing unittest tests.
**Why:** pytest collects `unittest.TestCase` natively, so the count is
identical (179 under both runners, measured). Rewriting working tests buys
nothing and risks weakening one.

## D-006 — Install all dependencies up front

**Chose:** every dependency the full system needs is installed and pinned in
Phase A.
**Rejected:** the old "install only in the sprint that needs it" rule.
**Why:** directive change. Sprints are retired and the phases run back to
back, so version drift between sprints is no longer the risk it was. Pins in
requirements.txt keep the environment reproducible.

## D-007 — Agent team: Opus everywhere, effort routed, main session still builds

**Chose:** ten project agents in `.claude/agents/`, all `model: opus`
(alias resolves to Opus 5.5 on this account per the docs check), `effort`
fixed per agent at medium, high or xhigh. Field names confirmed against the
Claude Code subagent docs before writing: `name`, `description`, `tools`,
`disallowedTools`, `model`, `effort`. Read-only agents (auditor, reviewer,
skill-scout) drop Write/Edit via `disallowedTools`.
**Rejected:** delegating every implementation step to `builder`.
**Why:** each subagent starts cold and re-derives context the main session
already holds. The independence rules are where delegation pays: audit,
review of crash-resume/approval/transaction/router diffs, and skill vetting
go to agents that did not write the code. Implementation that needs the
whole design in mind stays in the main session.

## D-008 — Run leases: pid is truth on the same host, staleness only across hosts

**Chose:** a `run_leases` row per run, owner `host:pid:random`, ownership
re-checked inside every step transaction with a rowcount check, plus a UNIQUE
(run_id, step_index) index as an independent backstop. On the same host a
live pid keeps the lease no matter how stale its heartbeat is; a dead pid is
taken over at once. Staleness (120s) decides only for an owner on another
host.
**Rejected:** heartbeat staleness alone (a single legitimate step can last
240s: two planner attempts at the 120s timeout, so any fixed window either
steals live runs or makes crash-resume wait); a background heartbeat thread
(sqlite connections are thread-affine, and it would keep a hung process's
lease alive forever).
**Why:** review H-2 showed the first version would hand a slow live run to a
second `--resume`. Access-denied from OpenProcess reads as alive (review M-1).

## D-009 — `importance` is written now, scored in Phase E

**Chose:** `remember_fact` accepts and stores `importance` (0-1) from Phase C
on; the recall scorer starts using it in Phase E step 6.9, with its own tests
replacing the test that pins its absence.
**Rejected:** keeping it NULL until "Sprint 5".
**Why:** ARGUS_BUILD.md 6.9 makes importance part of the build, and the
`write_fact` node assigns it. Storing it from the first write means no facts
exist without it when scoring turns on. The CLAUDE.md "stays NULL until
Sprint 5" line predates the directive (reviewer flagged the mismatch, M-4).

## D-010 — One migration file per concern

**Chose:** 003 run leases, 004 remember receipts, then separate files for the
graph tables, the model/learning tables, and the llm_calls widening.
**Rejected:** one large 003.
**Why:** each file stays transactional and reviewable on its own, and a
failure in one concern cannot block the others. The memory server can then
require exactly the files it reads (002, 004).

## D-011 — Migrations are serialised by an OS file lock

**Chose:** `apply_migrations` holds `<db>.migrate.lock` (msvcrt on Windows,
flock elsewhere) around the whole run, connect included, and also re-checks
`schema_migrations` inside each file's transaction.
**Rejected:** the in-transaction re-check alone.
**Why:** review H-1 measured two processes racing on the very first
`PRAGMA journal_mode = WAL` of a fresh file, before any transaction exists,
and the no-transaction path (001c) cannot be protected by a transaction.

## D-012 — Model pool from models already on disk; no engine upgrade mid-build

**Chose:** pool = qwen3.5:9b, phi4-mini (smallest chat model, the cheap
classifier/router candidate), llama3.1:8b (Meta family, for critique
independence from Qwen), gemma4:e4b (Google family; 9.6GB on disk, so the
profiler decides whether its offload makes it usable), nomic-embed-text.
**Rejected:** pulling gemma3:4b and qwen3.5:2b (both fail on Ollama 0.24.0:
"download the latest version"); upgrading Ollama to 0.40.x now;
deepseek-r1:8b (always-reasoning, think=False not guaranteed);
llama3.2-vision (7.8GB vision model, wrong job); `sunday` (custom, unknown
base).
**Why:** an engine upgrade invalidates every measurement in notes/ and the
profiler priors. The pool still covers four families and three sizes. The
upgrade is a question for Arya, not a silent change.

## D-013 — Retries are new nodes; critique passes facts through on PASS

**Chose:** a failed node gets a clone (attempt+1, `retry_root` = original)
wired to the same parents and children; inputs on one channel are satisfied
by the highest done attempt of one retry chain. `critique` provides the
`facts` channel only when its verdict is pass, so nothing downstream of
`extract` ever sees unreviewed facts; a rejected extract is retried as a new
(extract', critique') pair with the critic's reflection in the params.
**Rejected:** back edges for retries (breaks the DAG invariant and the
validator); letting resolve/diff consume extract output in parallel with the
critic (they would run on facts the critic later rejects).
**Why:** ARGUS_BUILD 6.5 requires a DAG. Routing data through the critic makes
"only reviewed facts are stored" structural.

## D-014 — Planner picks templates; it never emits raw graph structure

**Chose:** `plan`/`replan` output is a list of (template name, params);
the engine instantiates templates into nodes and edges and validates the
result before storing it.
**Rejected:** free-form node/edge lists from the model.
**Why:** small local models are reliable at "pick from a list and fill
fields" and unreliable at emitting DAGs. It also makes the planner's choices
countable, which is what procedural memory (template success rates) needs.

## D-015 — `search` is link search inside a fetched page, not a web search API

**Chose:** `search` ranks the links of an already fetched page by keyword
overlap and returns the best same-domain URL.
**Rejected:** a public search engine endpoint.
**Why:** constraint 3. A search query is target data sent to a third party;
following a link on the target's own site is not.

## D-016 — Diff is grounding-based; write_fact never rewrites content

**Chose:** a stored fact whose words are still on the page (verbatim after
whitespace/case folding, or 85% word overlap with every number present) is
`unchanged`, whatever wording the extractor used this time. `write_fact`
only scores importance; the stored content is always the page's own words.
**Rejected:** comparing the new extraction's text hash with the stored hash;
letting the writer model rephrase facts before storing.
**Why:** both make every weekly run report phantom "modified" facts, because
model wording drifts run to run even at temperature 0 across models. The page
is the ground truth that does not drift.

## D-017 — Thompson sampling over UCB

**Chose:** Beta posterior per (node type, model), seeded from the profiler's
quality with `prior_strength` pseudo-observations; Thompson sampling with a
greedy (posterior-mean) fallback when the exploration cap or low budget
forbids exploring. Per-node seeded RNG makes a decision reproducible.
**Rejected:** UCB1.
**Why:** a weekly system sees tens of pulls per arm per month; UCB's bonus is
tuned for many pulls and over-explores at this volume, while Thompson uses
the profiler prior directly and has a natural greedy fallback for the
bounded-exploration rule (ARGUS_BUILD 6.6).

## D-018 — Swap cost model

**Chose:** 0 if the model is resident; half its cold load if it can load
beside everything resident (co-residency matrix, both load orders must keep
both); otherwise its full cold load. Scaled by `latency_scale_s` and weighted
in config.
**Rejected:** the directive's literal "zero if co-resident".
**Why:** a co-resident model still has to load (a real 3-8s), it just evicts
nothing. Calling that free would make the router pick it too eagerly.

## D-019 — The approval gate is enforced three times

**Chose:** (1) the validator rejects any graph where an execute_action is
reachable without crossing an approval_gate; (2) readiness never readies an
execute_action until every gate ancestor is `done`, and fails closed when it
finds none; (3) the executor re-reads the approvals row and acts only on
`approved`, flips it to `executed` conditionally on `approved` in the same
transaction as the node's done, and the effect itself (outbox file named by
the idempotency key) is idempotent.
**Rejected:** any single one of these.
**Why:** hard constraint 4 must survive a bug in any one layer; each layer
has its own test and mutant.

## D-020 — resolve_entity is routed as an LLM node

**Chose:** normalized key match first, then memory-server semantic recall
(relevance >= 0.85 = same entity), then a model check (`entity_match`) only
for 0.60-0.85. The node runs in the GPU lane with a stored routing decision
because it may call a model; the path that decided each match is stored in
its output.
**Rejected:** a pure node that calls a model ad hoc (no stored decision, so
non-deterministic on resume); embedding in the orchestrator (Sprint 2 put
embeddings in the memory server).
**Why:** ARGUS_BUILD 6.9 and crash-resume determinism.

## D-021 — No tier labels in config; legacy roles map to a named baseline

**Chose:** `config/models.yaml` is an allowlist plus two named models:
`baseline` (the Sprint 1 single-model design, used by the legacy loop and by
Phase G's baseline) and `embed`. The `fast` tier and the `router: fast` role
are gone. `targets.yaml` holds only targets.
**Rejected:** keeping `chat`/`fast` keys.
**Why:** ARGUS_BUILD 6.1 forbids capability tiers in config or code; what a
model is good at is measured (`model_profiles`) and learned (`arm_stats`).

## D-022 — Critic independence: exclude the author, fall back only on an exhausted pool

**Chose:** a critique excludes the model that AUTHORED the judged output
(`_produced_by` on the subject node's output), not every model its decision
named. Only when every model in the pool is that author (a one-model pool,
or baseline mode) does the critic share it, and the decision's reason says so
("NO INDEPENDENT MODEL IN POOL"); the critic reward is never applied when
author == critic.
**Rejected:** failing or skipping the critique when no independent model
exists (a one-model install would never store a fact); excluding all decision
models (a cascade names two, which emptied a two-model pool and pushed the
critic onto the author; review High 2).
**Why:** independence is what makes a critic verdict evidence. Where it is
physically impossible the system still works, says so, and learns nothing
from it.

## D-023 — The executor claims the approval before acting

**Chose:** `execute_action` first flips the approvals row approved ->
executed with `{"claimed_by": node}` in one transaction, then performs the
idempotent effect (outbox file named by the key), then records the final
result with the node's done. A resumed node finishes its own claim; any
other state is refused.
**Rejected:** act-then-flip (a human could reject between the read and the
flip, or a crash could leave a sent message under a non-executed row;
review M10); adding an `executing` status (needs a table rebuild of
approvals for no extra safety).
**Why:** once claimed, nobody can change the decision under the action, and
every crash point (after claim, after effect) resumes to exactly one effect.
Proven by real-kill crash tests for both windows.

## D-024 — Step 12: the graph engine carries the legacy loop's guarantees; the loop stays as a reference

**Chose:** keep `orchestrator/loop.py` and its tests unchanged and running
(they guard the audited C1/H2 fixes), and prove every legacy crash-resume
guarantee on the graph engine with real-process kills: planner in flight
(`plan:0`), critic in flight (`critique:0`), plus the windows the loop never
had (extract, cross_model between samples, summarize, write_fact, remember
after commit, execute after claim and after effect). New runs go through the
graph engine (`python -m argus.graph.runner`).
**Rejected:** re-implementing the linear loop as a "legacy" graph template
just to make `scripts/crash_test.py` (which drives `orchestrator.runner`)
pass against the new engine.
**Why:** the guarantee is what must survive, not the state enum. A template
mimicking PLAN/ACT/OBSERVE would add a second orchestration path to keep
correct. `scripts/crash_test.py` still runs against the loop it was written
for.
