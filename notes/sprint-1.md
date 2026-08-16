# Sprint 1 — orchestrator

## Known limitations carried forward

### The planner is single-shot per run, not true ReAct

`_on_persist` routes `PERSIST -> PLAN` only when the local `task_queue` is
non-empty; `_on_plan`'s "queue is empty, ask the planner" branch only fires
when it's empty. Those two conditions are mutually exclusive, so once the
planner hands back a batch, that is the only batch for the run — a second
planner call is structurally unreachable, not just rare.

**Empirical evidence.** A stub planner was written that returns a task list on
its first call and `{"done": True, "tasks": []}` on any call after that. It
was called exactly once. Confirmed with `unittest.mock`-free direct
instrumentation (a `call_count` dict incremented inside the stub): after the
run completed, `call_count == 1`, even though the stub's own logic explicitly
handled being called a second time. The second-call branch never ran.

**Consequence.** Sprint 1 is plan-and-execute (plan once, work the list),
not interleaved reasoning and acting. A tool failure or a critique reflection
mid-run has no way to change what gets planned next — the plan was fixed at
the start.

**Why this is fine for now.** Deliberate for Sprint 1: the stub tools
(`fake_llm.py`) produce nothing worth re-planning against, so the gap costs
nothing yet.

**When to revisit.** Sprint 4, when real tool failures need an adaptive
response. Changing it means routing `PERSIST` on something other than
local-queue-length — e.g. an explicit "replan" signal set by `CRITIQUE` or
`OBSERVE`, checked alongside queue emptiness, rather than queue emptiness
alone deciding `PLAN` vs `DONE`.
