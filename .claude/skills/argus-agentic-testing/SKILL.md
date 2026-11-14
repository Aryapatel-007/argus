---
name: argus-agentic-testing
description: How to test Argus's agentic behaviour - graph trajectories, routing decisions, strategies, crash-resume with real process kills, and negative controls - without a live model where possible and with one where it matters. Use when writing or reviewing tests for the graph engine, the router, the scheduler, the legacy loop, or the eval harness.
effort: high
---

# Testing Argus's agentic workflows

Written in-repo in Phase D: the external candidates assumed hosted judges,
API keys, or argued against asserting trajectories. Argus needs trajectory
assertions, because the approval gate and crash-resume are properties of the
path, not only of the final answer.

## Rules

1. **Every new guarantee gets a negative control.** Break the code on purpose
   (monkeypatch, or a mutant in `scripts/mutation_check.py`) and show the test
   fails. Add a mutant for each guarantee that protects a hard constraint.
2. **Fake the model at the one seam, not above it.** Graph engine:
   inject a fake `ModelBackend` into `Services`. Legacy loop: patch
   `llm._invoke`. Everything above the seam (prompt rendering, schema
   validation, llm_calls logging, routing) runs for real.
3. **Assert the trajectory, not only the outcome.** Read `node_events`,
   `routing_decisions`, `llm_calls` and `approvals` back from SQLite and
   assert order and counts: which nodes ran, in what status order, with which
   model, how many swaps, whether any action ran before its gate was approved.
4. **Crash tests kill real processes.** `subprocess.Popen` the runner, wait
   for a trigger row (e.g. a routing decision stored but no output yet), kill
   it, resume, and compare with a control run. On Windows also list the
   process tree afterwards (no orphaned MCP servers).
5. **Determinism.** Fakes return the same output for the same (node, model,
   sample index). Resume must reuse stored routing decisions and stored
   samples; assert byte equality of stored outputs between a crashed-and-
   resumed run and a control run.
6. **Live-model tests are few and marked.** `@pytest.mark.ollama`. They check
   that real models produce schema-valid output for each node type, not exact
   wording. The suite must pass with Ollama down except for those.
7. **Property tests on structure.** Use hypothesis for validator, readiness
   and retry-wiring invariants (see the `property-based-testing` skill).

## Trajectory assertions worth having

- No `execute_action` reaches `running` while its gate is not `done`.
- Every LLM node that reached `running` has exactly one non-superseded,
  frozen `routing_decisions` row created before its start event.
- No critique node's model equals the model of the node it judged.
- Resume after a kill: no node output changes; no approval row duplicates.
- Swaps per run = number of model changes between consecutive GPU-lane
  calls; the router with swap cost must not exceed the baseline's.
