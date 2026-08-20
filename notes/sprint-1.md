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

## Findings — real Ollama runs

### think=False does not degrade planning quality

Real Ollama runs confirm `think=False` does NOT degrade planning quality. The
`thought` field was coherent in every sample.

### What looked like model flakiness was a deterministic prompt bug

5/5 first-attempt planner calls failed schema validation, always with the same
two errors (`tasks.0.tool_name` missing, `done` missing), always at
`eval_count=110`. 5/5 retries then succeeded, always at `eval_count=116`.
Byte-identical failure shape every time — not sampling noise.

**Root cause.** `planner.md` never named `tool_name` literally, and rendered
the tool catalog as a call signature — `stub_fetch_ok(url: str)` — which
primes `{"tool": ...}` as the natural shape. Separately, `done` appeared only
inside a conditional that was false on the normal path, so nothing instructed
emitting the key at all.

**After the prompt fix:** 5/5 first-attempt successes, zero retries,
`eval_count=88` on every run. Shorter and directly correct.

### tok/s across all real runs: 12.3 to 23.6

No collapse, no runaway. The 3.9 tok/s cliff from the Sprint 0 probe did not
recur under this workload. No instability observed.

### Untested lever, recorded not chased

`PlannerOutput.model_json_schema()` emits `required: ["thought", "done"]`
only, with `tasks.items` as a `$ref` into `$defs`. Flattening `Action` inline
remains available if grammar-constrained decoding proves weak later. Not
needed this sprint — the prompt fix alone resolved it.

## Closed from CLAUDE.md's Sprint 1 open-items list

Moved out of CLAUDE.md once resolved, so the file prepended to every turn
doesn't keep paying a token cost for history nobody needs. Full detail here
instead.

### KV cache quantization — tested, closed

`OLLAMA_KV_CACHE_TYPE=q8_0` with `OLLAMA_FLASH_ATTENTION=1` applies correctly
(confirmed via `KvCacheType:q8_0` in the load request and
`kv cache device=CUDA0 size="1.3 GiB"` in the server log, down from 1.4GB at
f16). Total memory only dropped 8.2GB -> 8.1GB — still over the 7.6GB
ceiling, still 32/33 layers on GPU. The 2.2GB output layer on CPU is the real
bottleneck, untouched by KV cache size.

`q4_0` not worth trying: even its full theoretical saving wouldn't close the
remaining 0.6GB gap, and it costs measurable quality versus q8_0's near-zero
cost. Keeping q8_0 as a free, harmless setting — not revisiting this lever
again.

## Crash-resume — verified against real Ollama

`scripts/crash_test.py` kills a real OS process mid-run against a real Ollama
call, then resumes it, and compares the result to an uninterrupted control
run. Three scenarios:

### kill-during-planner-call — PASS

On target. 24/24 checks. Mid-LLM-call confirmed via `llm_calls`-row polling
(the row is written only after a call returns, so an unmoved count means the
call is still in flight). Ungracefully killed, proven three ways: `poll()` was
`None` immediately before the kill, the subprocess returncode matched a hard
kill, and the last checkpoint before the kill was non-terminal. Resumed to
DONE. `facts_staged` content matched the control run byte-for-byte (task_id
excepted — see harness bug #1 below).

### kill-during-critic-call — PASS

On target. 24/24 checks. Same guarantees as above.

### kill-between-steps — INCONCLUSIVE

Could not land the window in 5 attempts. Measured on this machine: non-LLM
step gaps ~1ms, the open-transaction window (`llm_calls` row committed ->
checkpoint committed) ~0-6ms, against ~13s of model-call time per run. The
window is narrower than the kill latency achievable from a separate process.

Not a gap in the guarantee. This mode — a crash during a local SQL write
inside `db.atomic()` — is already proven by two committed tests:
`test_storage.py`'s `test_failure_mid_transaction_writes_nothing` (during
development, verified not vacuous via a one-off negative control: a
deliberately broken `atomic()` that COMMITs on exception instead of rolling
back was shown to fail this test), and `test_loop.py`'s simulated
mid-transaction crash tests (`raise` inside a `with db.transaction` block).
The real-process harness confirms the harder case, a crash mid external call;
the easier case was already proven by a method suited to it — you cannot
reliably OS-kill a process inside a 1ms window, but you can trivially raise an
exception at an exact line.

### Two harness bugs found and fixed during this work

1. **Equivalence check was unsatisfiable by construction.** It compared full
   `facts_staged` dicts, including `task_id`, which embeds `run_id`
   (`f"{run_id}:task-{n}"`). Two distinct runs always have different
   `run_id`s, so no crash-resume behavior could ever make this pass — it
   failed even when nothing was wrong. Fixed by normalizing each side's
   `run_id` to a placeholder before comparing, so a fact attached to the wrong
   task *ordinal* is still caught. Verified with two negative controls: a real
   content difference and a task-ordinal mismatch both still produce a
   failure after normalization.
2. **Between-steps targeting polled too slowly for the window it was aiming
   at.** A 50ms poll interval against a <10ms window meant the trigger
   checkpoint was detected only after the process had already moved on into
   the next model call. Fixed with a tight poll (0.2ms, persistent connection)
   on the `llm_calls` row landing, plus 5 retries and an explicit
   `INCONCLUSIVE` report on a miss rather than silently re-testing the
   mid-LLM-call mode a third time.
