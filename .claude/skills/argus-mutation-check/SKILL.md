---
name: argus-mutation-check
description: Mutation-style check that Argus's tests actually guard its critical guarantees (approval gate, leases, BEGIN IMMEDIATE, think=False, checkpoint atomicity, idempotent memory writes, routing determinism). Use after changing any guarded code or its tests, and before each phase commit.
effort: medium
---

# Mutation check

```powershell
python scripts/mutation_check.py              # every mutant
python scripts/mutation_check.py gate lease   # only matching names
```

Each mutant copies the repo to a temp dir (the working tree is never
touched), plants one known bug, and runs only the tests meant to catch it.

- `KILLED` is good. `SURVIVED` means those tests are vacuous for that bug:
  strengthen the test, do not delete the mutant.
- `ANCHOR-MISSING` means the guarded code moved. Update the mutant's
  `old` text to the new code, keep the same bug.

When you add a guarantee that protects a hard constraint, add a mutant for it
in `MUTANTS`. First real catch (Phase D): `deferred-begin` survived because
the `atomic()` test wrote without reading first, so a plain BEGIN still
waited on the lock. Fixed by a read-then-write test.

Why not a mutation tool: Trail of Bits' mutation skill drives mewt, which has
no Python support and does not run on Windows.
