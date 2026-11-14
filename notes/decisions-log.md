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
