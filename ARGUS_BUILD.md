# ARGUS_BUILD.md: End-to-End Build Directive

You are Claude Code, working in the root of the Argus repository.
This file is your complete brief for finishing Argus.
Read all of it before doing anything.
Sprints are retired. This file replaces them with phases that you run yourself.

---

## 0. How this file works

This build will not finish in one session.
You will hit context limits and usage limits along the way.
So the build must be resumable from disk, not from your memory.

Rules for that:

- Keep a file called `BUILD_STATE.md` in the repo root.
- At the start of every session, read `BUILD_STATE.md` first, then `CLAUDE.md`, then this file.
- If `BUILD_STATE.md` does not exist, create it and start at Phase A.
- After every completed step, update `BUILD_STATE.md`: current phase, current step, what is done, what is next, anything blocked.
- Write it as if a stranger will resume from it with no other context. That stranger is you, next session.
- Commit `BUILD_STATE.md` together with the work it describes.

---

## 1. Authority: what you decide, what you never change, when you stop

### 1.1 Hard constraints. These beat everything in this file.

These come from `CLAUDE.md` and are not negotiable:

1. Everything runs locally at runtime. Ollama with local models only. No paid APIs, no hosted inference.
2. Zero recurring cost.
3. All data stays on the machine. No telemetry, no external logging.
4. Nothing externally visible happens without human approval. An approvals row is written and execution of that action blocks. This is structural, not a confidence threshold.

If anything in this file seems to conflict with these, the constraints win and you write the conflict into `BUILD_STATE.md`.

### 1.2 Settled findings you must not rediscover or undo

Read `CLAUDE.md` and `notes/` for the full list. The ones most likely to be broken by accident:

- `think` is a top-level field on Ollama chat and generate calls, not inside `options`. Pass it explicitly on every call. On `qwen3.5:9b`, `think="low"` is silently coerced to on. With thinking on, generation is unbounded and was observed returning empty content after 26 minutes.
- Every Ollama call has a hard 120s timeout and logs tok/s to `llm_calls`, including timed-out calls.
- Every write goes through `db.atomic()`, which issues `BEGIN IMMEDIATE`. Never add a second write helper.
- Never hold a transaction open across an Ollama call. Compute first, then write.
- Migration state lives in `schema_migrations`. `PRAGMA user_version` is not used.
- `nomic-embed-text` outputs 768 dimensions. That fixes the `facts_vec` width.
- The state transition and its checkpoint go in one transaction. Crash-resume must keep working at every layer you add.
- Staged outputs are checkpointed before any side effect, and reused on resume, never regenerated.
- Hand-built orchestration. No LangGraph, no CrewAI, no AutoGen, no agent framework. Libraries for narrow jobs (MCP SDK, sqlite-vec, Browser Use, APScheduler) are fine.
- Existing negative-control tests stay. Never delete or weaken a test to make something pass.

### 1.3 What you are now allowed to decide alone

The old rule in `CLAUDE.md` said design questions go to a separate architect chat.
That rule is replaced by this one:

- You make design decisions yourself.
- Every real decision gets an entry in `design-decisions.md` style: what you chose, what you rejected, and why. Put new entries in `notes/decisions-log.md` and keep them numbered.
- Update the "How to work with me" section of `CLAUDE.md` to reflect this, so future sessions are not blocked by the old rule.

### 1.4 When you must stop and wait for Arya

Write `BLOCKED: needs Arya` at the top of `BUILD_STATE.md`, explain in plain sentences, and stop, when:

- A step would violate a hard constraint.
- A step would delete or overwrite `data/argus.db` or any real collected data. Test databases are fine.
- A step needs a force push, history rewrite, or deleting a branch.
- A skill or dependency asks for credentials, an API key, a paid account, or sends data off the machine at runtime.
- Free disk space drops below 15GB before pulling models.
- The Phase G baseline comparison shows the graph engine is worse than the single-model baseline and you cannot fix it.

Everything else, you decide, log, and continue.

---

## 2. Phase A: Environment and dependencies

Goal: every dependency the full system needs is installed and verified once, here, at the start.
This replaces the old "install only in the sprint that needs it" convention. Update `CLAUDE.md` to say so.

Environment facts: Windows 11, PowerShell 7.6, Python 3.11.9, `.venv` at repo root, `python` not `python3`, RTX 4060 8GB with about 7.6GB usable, 16GB RAM.

Steps:

1. Activate `.venv`. Confirm with `Get-Command python` that it resolves inside `.venv`.
2. Read `requirements.txt` and what is already installed. Do not downgrade anything that tests depend on.
3. Install what the remaining phases need. At minimum check and add:
   - `mcp` (v2, already decided in Sprint 2)
   - `sqlite-vec` (already 0.1.9, keep)
   - `browser-use` plus its Playwright browser install
   - `playwright` browsers via `python -m playwright install chromium`
   - `apscheduler`
   - `httpx`
   - a dashboard stack: pick FastAPI plus a minimal server, or Streamlit. Decide, log the decision.
   - test tooling: `pytest`, `pytest-asyncio`, `hypothesis` for property tests on graph invariants
   - `deepeval` only if it can run fully against local Ollama models with no API key. Verify that before installing. If it needs a hosted key, skip it and log why.
4. Pin versions in `requirements.txt`. Add a `requirements-dev.txt` for test-only tools.
5. Verify each install with a one-line import check. Record versions in `notes/environment.md`.
6. Check Browser Use specifically: confirm it can run against a local Ollama model and does not require a hosted LLM key. Check its current docs. If it cannot, log it and fall back to raw Playwright plus your own LLM extraction step.
7. Check `ollama --version`, list local models with `ollama list`, and record free disk space.
8. Run the full existing test suite. Record the passing count in `BUILD_STATE.md`. This number must never go down.

Commit: `Phase A: environment and dependencies`.

---

## 3. Phase B: Build your agent team before any real work

Goal: a team of subagents in `.claude/agents/`, each with one job, so the main session coordinates and delegates instead of doing everything itself.

### 3.1 Verify the format first

Before writing any agent file, use the `claude-code-guide` agent to confirm the current subagent file format: frontmatter fields, how `model` is specified, and the exact field name for reasoning effort.
Do not guess field names. A misspelled effort field silently does nothing.

### 3.2 Model and effort policy

- Every agent runs on Opus. Confirm with `/model` that the `opus` alias resolves to Opus 5.5 on this account. If it needs a pinned id, use exactly what `/model` shows.
- Effort floor is `medium`. No agent ever runs below Opus medium. No Sonnet, no Haiku.
- Allowed effort levels: `medium`, `high`, `xhigh`.

### 3.3 About the effort-switching agent

A subagent cannot change the main session's model or effort while it runs.
So effort switching is done by routing, not by flipping a setting:

- Each worker agent has its effort fixed in its own frontmatter.
- The `effort-governor` agent reads a task and decides which worker, at which effort, should take it.
- The main session always asks the governor before dispatching non-trivial work, then dispatches to the agent it names.

### 3.4 The team

Create these files in `.claude/agents/`. Write each one's system prompt properly: its job, what it must never do, what it returns.

| Agent | Effort | Job | Writes code? |
|---|---|---|---|
| `effort-governor` | medium | Classifies a task by risk and difficulty, returns which agent should do it and why. Escalates to xhigh for anything touching crash-resume, concurrency, the approval gate, or the router. | No |
| `auditor` | high | Read-only review of existing code against `CLAUDE.md`. Finds flaws, gaps, untested claims. | No |
| `architect` | xhigh | Designs the graph engine, schemas, invariants. Produces specs other agents build against. | Specs only |
| `builder` | high | Implements modules against an agreed spec. | Yes |
| `builder-routine` | medium | Dependencies, boilerplate, docs, config, mechanical edits. | Yes |
| `test-engineer` | high | Tests, property tests, negative controls that prove tests are not vacuous. | Yes |
| `debugger` | xhigh | Failures nobody understands yet. Reproduce first, then fix. | Yes |
| `skill-scout` | medium | Finds, reads and vets external skills. Never installs without a full read. | No |
| `reviewer` | high | Independent review of a diff before commit. Has not seen the work being produced. | No |
| `doc-keeper` | medium | Keeps `BUILD_STATE.md`, `CLAUDE.md`, `notes/`, and the decisions log current. | Docs only |

Rules for how they are used:

- No agent reviews its own work. `reviewer` checks `builder` output, `test-engineer` writes tests for code it did not write where possible.
- Anything touching crash-resume, the approval gate, transactions, or the router gets a `reviewer` pass before commit. No exceptions.
- Subagent reports are not shown to Arya. The main session relays what matters into `BUILD_STATE.md`.

Commit: `Phase B: agent team`.

---

## 4. Phase C: Study what exists and find what is wrong

Goal: a written audit before anything new is built on top of the old.

Dispatch `auditor`. It reads every file under `src/`, `tests/`, `scripts/`, `notes/`, plus `CLAUDE.md`.

It produces `notes/audit.md` with these sections:

1. **Map.** What each module does, in a short paragraph each. How they connect.
2. **Claims without proof.** Any guarantee stated in a docstring, comment or note that no test actually exercises.
3. **Known open items.** From `CLAUDE.md` and `notes/`: writer quality probe, single-shot planner, orchestrator has no MCP client, crash-resume across the subprocess boundary, Windows orphan processes, entity_key drift, co-residency once a browser is running, importance scoring.
4. **Flaws.** Bugs, races, layering problems, error paths that crash instead of degrade.
5. **What the graph engine will break.** Which current assumptions stop being true when a single linear loop becomes a graph of many models. Be specific: the checkpoint payload shape, `llm_calls.role`, the single-model config, the loop's state enum.
6. **Recommended fixes**, ranked by severity.

Then the main session fixes everything ranked critical or high before Phase E starts, each fix with a test.
Medium and low items go into `BUILD_STATE.md` as a backlog.

Commit: `Phase C: audit and critical fixes`.

---

## 5. Phase D: Skills for testing agentic workflows

Goal: bring in external skills that make testing and auditing this system better.

Dispatch `skill-scout`.

Arya named `cloudflare/security-audit-skill` as an example.
First verify whether that repository actually exists and what it actually does. Do not assume.
Then search GitHub and skill directories for skills covering:

- security auditing of agent code (prompt injection paths, unsafe tool use, secrets in code)
- testing LLM and agentic workflows (trajectory tests, tool-call correctness, eval harness patterns)
- Python test quality (property testing, mutation-style checks)
- MCP server testing

Vetting rules, all mandatory:

1. Read the entire skill, every file, before installing. A skill runs with your permissions.
2. Reject any skill that sends repo contents off the machine, needs an API key or paid account, or runs network calls you cannot explain.
3. Reject skills that are abandoned with known open security issues.
4. Prefer installing at project scope so it lives in the repo.
5. For each skill installed, write one paragraph in `notes/skills.md`: what it is, where it came from, the commit you installed, why you trust it, what you will use it for.
6. For each skill rejected, one line on why.

Then actually use them: run the security audit skill across the codebase now, and fix anything critical.
Run it again at the end of Phase F.

If nothing suitable exists for a category, write your own small project skill for it under `.claude/skills/` and log that you did.

Commit: `Phase D: testing and audit skills`.

---

## 6. Phase E: The graph workflow engine

This is the core of the whole build. Use `architect` at xhigh for the design, `reviewer` on every piece, and `debugger` at xhigh when something fails that you do not understand.

### 6.1 The problem being solved

The current orchestrator is a single linear loop on one model: plan once, then act, observe, critique per task.
That design has three real limits:

- The planner is single-shot. It cannot react to what it learns mid-run.
- Every role is bound to one model in config. Nothing adapts.
- Work that could overlap, like a page load and an unrelated model call, runs strictly in sequence.

The replacement is a dynamic, typed task graph, executed by a scheduler that decides at runtime which local model handles each node, and learns from outcomes which choices work.

There are no fixed tiers. No "fast, medium, smart" labels anywhere in config or code. A model's suitability for a job is a measured, per-task-type number that changes as evidence arrives.

### 6.2 The constraint that makes this genuinely hard

On this machine only one chat model fits comfortably in VRAM at a time.
Swapping models costs about 6 seconds cold load.
`qwen3.5:9b` and `nomic-embed-text` were measured as co-resident. Other pairs are unknown.

So routing is not "pick the best model per node."
It is a scheduling problem: pick models per node so that total quality is high, total time is low, and the number of model swaps is minimized across the whole graph.
A router that ignores swap cost will thrash and be slower than one model.
This is the central difficulty. Design around it explicitly.

### 6.3 Model pool and capability profiles

**Pool.** Choose 4 to 6 local models that each fit under the VRAM ceiling on their own, chosen for diversity, not size alone. Include at least:

- the current `qwen3.5:9b`
- `phi4-mini`, already pulled
- a different model family for critique independence
- a small model suited to cheap classification and routing
- `nomic-embed-text` for embeddings

Check the Ollama library for current tags. Check sizes against the VRAM ceiling and disk before pulling. Log each choice and why.

**Discovery.** The pool is not hardcoded. A `model_registry` module reads `ollama list`, joins it with `config/models.yaml` (which only says which models are allowed, never what they are good at), and with measured profiles from the database.

**Profiler.** Build `scripts/profile_models.py`. For each model in the pool, run a fixed battery across every node type:

- measured: VRAM footprint, GPU/CPU split, cold load seconds, warm load seconds, tok/s
- think support: whether `think=False` is honored, whether levels are coerced
- structured output: JSON validity rate against each node type's schema, over repeated runs
- task quality: a small labeled fixture set per node type, scored
- stability: run-to-run variance of key fields, like `entity_key`
- co-residency: for each pair, whether both stay loaded together, measured via `ollama ps`

Store results in a `model_profiles` table. These become the router's priors. Re-run the profiler whenever the pool changes.

### 6.4 The graph

**Node.** Every unit of work is a typed node. Each node type declares:

- an input schema and an output schema
- whether it needs an LLM, a tool, or neither
- whether it has an external side effect
- its validator: deterministic checks that run before any critic
- its default execution strategy

Starting node types, extend as needed:

- `plan` and `replan`: produce or extend the graph itself
- `search`, `fetch_page`: tool nodes, I/O bound
- `extract`: structured facts from page text
- `resolve_entity`: decide whether an extracted item is a known entity
- `recall`: query the memory server
- `diff`: compare against memory
- `critique`: verdict, confidence, reflection
- `write_fact`: produce the durable fact
- `remember`: persist through the memory server over MCP
- `summarize`: digest prose
- `propose_action`: draft something external
- `approval_gate`: writes the approvals row and blocks
- `execute_action`: only reachable after an approved gate
- `reflect`: consolidate many memories into higher-level ones

**Edges** carry data dependencies. A node is ready when all its inputs are satisfied.

**Dynamic expansion.** The graph is not fixed at plan time. `plan` emits an initial graph. Any node can return an expansion request, and a `replan` node inserts new nodes based on what was actually observed. This replaces the single-shot planner and is how ReAct-style adaptation works at graph scale.

**Subgraph templates.** Reusable patterns, for example "check a careers page" as fetch, extract, resolve, diff, critique, write, remember. The planner instantiates templates instead of inventing structure every time. When a run's subgraph succeeds well, store it as a template candidate. This is procedural memory: the system learns how to do things, not only what it saw.

### 6.5 Structural invariants

Before any graph or expansion executes, a validator checks it. If it fails, nothing runs.

- It is a DAG. No cycles. Retries are new nodes, not back edges.
- Every `execute_action` node has an `approval_gate` ancestor on every path, and that gate must be in approved state before the node becomes ready. The approval gate is enforced by graph structure, not by trusting the planner.
- No node with an external side effect lacks an idempotency key, generated and stored before it runs. The key lives on `approvals`, as decided in Sprint 2.
- Hard limits: maximum nodes per run, maximum depth, maximum expansions, maximum retries per node. A planner can never grow the graph forever.
- Every node's input schema is satisfiable by its parents' output schemas.

Write property-based tests with `hypothesis` that generate random graphs and expansions and prove the validator rejects every bypass of the approval gate and every cycle. Add a negative control: a deliberately broken validator must let a bypass through, proving the test is not vacuous.

### 6.6 The router

For each ready LLM node, the router chooses a model and an execution strategy.

**Inputs:**

- the node type and its difficulty signal from the planner
- each candidate model's profile for that node type
- learned statistics for each (node type, model) pair
- which model is currently resident
- the co-residency matrix
- the run's remaining time and token budget
- each model's current health

**Scoring.** For each candidate model, compute:

- expected quality for this node type, from the learned posterior
- expected latency, from tok/s and expected output length
- swap cost, zero if resident or co-resident, otherwise measured cold load
- failure risk, from JSON validity rate and timeout history

Combine into one utility with weights in config. Do not hardcode the weights in source.
Log every candidate's score breakdown, not just the winner. Sprint 6-style observability needs to answer "why did it pick this model" from the database alone.

**Learning.** Treat each (node type, model) pair as an arm. Use Thompson sampling or UCB. Pick one, justify it in the decisions log. Reward comes from:

- deterministic validator pass or fail
- critic verdict, only when the critic is a different model from the one judged, so a model never grades itself
- later evidence, such as a fact that the eval harness marks correct or wrong

Keep exploration bounded so a production run never spends much of its budget on experiments.

**Execution strategies** the router can choose per node:

- `single`: one call
- `self_consistency`: N samples from one model, take the agreeing answer, use disagreement as an uncertainty signal
- `cascade`: try the cheapest adequate model first, escalate to a stronger one only if the validator or critic rejects
- `cross_model`: two different models must agree. Required for every `propose_action` node and for any fact whose confidence falls below threshold
- `reflexion_retry`: on failure, the critic's written reflection is fed into the retry

**Determinism for crash-resume.** A routing decision is written to `routing_decisions` and checkpointed before the call runs. On resume, the stored decision is reused, never recomputed. The learned statistics will have changed by then, and re-routing on resume would make a crashed run behave differently from one that never crashed.

### 6.7 The scheduler

**Two lanes.**

- A GPU lane with exactly one slot. All LLM calls queue here.
- An I/O lane with several slots, using asyncio. Browser fetches, MCP calls and searches run here concurrently.

This lets a page load overlap a model call, which the linear loop never could.

**Swap-aware batching.** When several LLM nodes are ready at once, group them by their routed model and drain the group for the resident model before swapping. Allow the router to revise a not-yet-started routing choice if doing so avoids a swap and costs little quality, and log every such revision.

**Health and circuit breakers.** Watch tok/s from `llm_calls`. Throughput has been observed collapsing about 5x under VRAM pressure mid-session. If a model's tok/s drops sharply, or it times out repeatedly, mark it degraded and stop routing to it until a health check passes.

**Budgets.** Every run has a time and token budget. As budget runs low, the scheduler prunes optional branches first and never prunes an approval gate.

### 6.8 State, checkpoints and schema

The Sprint 1 guarantee must survive: a crash at any point resumes from the last completed step, with no duplicated side effects and no regenerated outputs.

At graph scale that means:

- Every node state change is checkpointed in the same transaction as the change itself.
- Node outputs are stored before any downstream node consumes them.
- Expansions are stored before their new nodes can run.
- Routing decisions are stored before their calls run.

Add migration 003 through the existing migration system. Likely tables: `graphs`, `graph_nodes`, `graph_edges`, `routing_decisions`, `model_profiles`, `arm_stats`, `templates`. Widen `llm_calls` to reference a node, and record which model, which strategy, and which sample index.
Keep it one migration file per concern if that is cleaner, and log the choice.

### 6.9 Memory integration

- `recall` uses the existing hybrid retrieval in the memory server.
- Add importance scoring, which `CLAUDE.md` deferred. Importance is assigned by a model at write time and fed into the scorer alongside relevance and recency.
- Add a scheduled `reflect` graph that clusters related facts per target and writes higher-level summary facts that are themselves retrievable.
- `resolve_entity` moves beyond exact `entity_key` match: normalized key first, then embedding similarity above a threshold, then a model check for ambiguous cases. Log which path decided each match.

### 6.10 Build order inside Phase E

Do these in order, committing each with tests:

1. Node type registry and schemas
2. Graph data model plus migration 003
3. Validator plus property tests plus negative control
4. Model registry plus profiler, then run the profiler on the real pool
5. Router scoring with fixed priors, no learning yet
6. Scheduler with both lanes and swap-aware batching
7. Checkpointing and resume at graph level, with a crash test
8. Execution strategies
9. Learning: arm statistics and the bandit
10. Dynamic expansion and replanning
11. Templates and procedural memory
12. Port the old linear loop's behavior onto the graph engine, so the existing crash tests still pass against it

---

## 7. Phase F: The rest of the product

Build these on top of the graph engine. Each one is a set of node types, tools and tests, not a separate system.

1. **Orchestrator to memory server over MCP.** Spawn per run with `--run-id`, after the runs row is committed. Real stdio boundary, never import the backend directly. Clean shutdown. Fresh spawn on resume.
2. **Crash-resume across the subprocess boundary.** Kill the orchestrator while `remember` is in flight. Prove no duplicate facts, no phantom "modified" facts, staged output reused byte for byte, and no orphaned memory server process on Windows. Check by real process listing, do not assume stdin EOF handles it.
3. **Browser tool server.** Second MCP server, Browser Use on Playwright, degrading gracefully: partial result plus a confidence flag, never an unhandled exception that kills a run. Real wait conditions, never fixed sleeps. Close contexts and pages. Self-throttle per domain. After this exists, re-measure co-residency with the browser running.
4. **Watch logic.** Targets from `config/targets.yaml` become graphs. Diff against memory. A digest of only what genuinely changed. Silence is a valid result.
5. **Approval flow.** Approvals as rows. A way for Arya to approve or reject, from the dashboard and from the command line. Approved actions execute exactly once.
6. **Observability.** Every node a span, every run a trace. Routing decisions with score breakdowns. A local dashboard: run history, trace tree per run, routing explanations, pending approvals, model health.
7. **Evaluation harness.** 20 to 30 scenarios against fixture pages served locally that you control. Score correctness, faithfulness, and approval-gate bypass attempts. Score at three levels: end to end, trajectory, and per node. The judge must be a different model from the one being judged. Calibrate the judge against a small hand-labeled set, record agreement.
8. **Scheduling.** APScheduler in process. A real hard-kill mid-run and resume test. A memory consolidation policy.
9. **Polish.** One command to start everything. A README that explains the architecture in the first two minutes. A demo script.

---

## 8. Phase G: Prove the complexity earns its place

This is the one rule that protects the whole design from being complexity for show.

Run the evaluation harness three ways on identical scenarios:

1. **Baseline:** everything on `qwen3.5:9b`, single strategy, no learning. This is the Sprint 1 design.
2. **Graph, fixed routing:** graph engine with profiler priors, learning off.
3. **Graph, learning:** full engine, after a warm-up of runs.

Compare quality, wall-clock time, model swaps per run, timeouts, and approval-gate violations, which must be zero in all three.

Write `notes/routing-results.md` with the numbers and an honest reading of them.

- If the learning engine beats the baseline, say by how much and where.
- If it does not, find out why with `debugger` at xhigh and fix it.
- If it still does not beat the baseline after real effort, do not hide it. Stop with `BLOCKED: needs Arya` and explain.

An interviewer will ask "did the complicated version actually help." This file exists so the answer is a number.

---

## 9. Definition of done

Argus is done when all of these are true:

- A scheduled run checks every target in `config/targets.yaml` unattended and produces a digest of genuine changes.
- Nothing external happens without an approved approvals row, and a property test proves the graph cannot bypass the gate.
- A hard kill at any point resumes cleanly, proven by real-process crash tests for the graph engine, the memory server boundary, and the browser server.
- Routing decisions are explainable from the database alone.
- The dashboard shows runs, traces, routing explanations, approvals and model health.
- Phase G numbers are written down honestly.
- The security audit skill reports nothing critical.
- The full test suite passes, and its count never went down at any point.
- `README.md`, `CLAUDE.md`, `notes/decisions-log.md` and `notes/routing-results.md` are current.

---

## 10. Working rules for every phase

- Run the full test suite before every commit. Record the count in `BUILD_STATE.md`.
- Small commits with clear messages, pushed after each completed step.
- New guarantees get a negative control: a deliberately broken version that the test must catch.
- Measurements go in `notes/`. Decisions go in `notes/decisions-log.md`. Current truth goes in `CLAUDE.md`.
- Keep `CLAUDE.md` under 250 lines. When it grows, move detail to `notes/` and leave a pointer.
- Use `/clear` between unrelated phases and `/compact` inside long ones. `BUILD_STATE.md` is what survives, not the conversation.
- Windows specifics: PowerShell syntax, `python` not `python3`, and remember child processes are not killed with their parent.
- Never trust a remembered number over a live command. If two facts disagree, run the real check, then write the answer down once in the right place.

Start now: read `BUILD_STATE.md` if it exists, otherwise create it and begin Phase A.
