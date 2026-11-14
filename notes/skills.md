# Skills (Phase D, 2026-11-14)

Vetting was done by the skill-scout role (read every file, record the commit,
reject anything that phones home or needs a key). The main session then read
every file of the one install again before copying it.

## Installed

**property-based-testing** (`.claude/skills/property-based-testing/`).
Trail of Bits, https://github.com/trailofbits/skills,
`plugins/property-based-testing/skills/property-based-testing/` at commit
`149d7b21f4067765b82e67029730b42d228ef4b4`, CC-BY-SA-4.0. Six Markdown
files, no scripts, no network use, no keys, from an active security firm's
maintained repo. One edit: frontmatter `effort: low` raised to `medium` (the
build's effort floor); recorded in its PROVENANCE.md. Used for writing and
reviewing the hypothesis suites: graph validator invariants, readiness and
retry wiring, router determinism, and spotting vacuous `assume()` or
tautological properties.

## Written in-repo (nothing suitable existed)

- **argus-security-audit** — scanner (`scripts/security_scan.py`, tested by
  `tests/test_security_scan.py`) plus a 10-point checklist for what a regex
  cannot see (gate structure, prompt injection from page text, memory
  poisoning, Windows process control).
- **argus-agentic-testing** — trajectory assertions from SQLite, fake model
  at one seam, real-process crash tests, negative controls.
- **argus-mcp-testing** — backend, in-memory Client, real stdio subprocess;
  stdout discipline, idempotency, Windows orphan checks.
- **argus-mutation-check** — `scripts/mutation_check.py`, hand-written
  mutants for each hard-constraint guarantee, run in a temp copy of the repo.

## Rejected

- **cloudflare/security-audit-skill** (exists, official, commit `c1c8a8c1`):
  its validators throw on Windows (no `O_NOFOLLOW`, open issues #53, #117);
  open issue #128 reports it wiping a Windows 11 user profile (`~/.claude`,
  `~/.ssh`) when run without a sandbox, unanswered; #148 target-config
  loading issue; no merges in 60+ days. Its `AI-AND-LLM.md` is good reading
  material, not installed.
- **trailofbits mutation-testing** (`123037ec`): drives mewt, which has no
  Python support and does not run on Windows.
- **Agent-Skills-for-Context-Engineering / evaluation** (`8fc51c7c`):
  generic, one-person repo, advice assumes hosted judges and argues against
  trajectory assertions.
- **anthropics/skills mcp-builder** (`ca45fc49`): its evaluation script
  calls the paid Anthropic API with tool outputs; teaches MCP SDK v1.
- **ckorhonen mcp-tester** (`53fe880b`): low-reputation, stale, manual
  probing only, enforces kebab-case tool names.

## Runs

- Security scan, end of Phase D: 0 blocking findings after fixing
  `scripts/smoke_test.py` (no think=False, no timeout; audit L14) and
  allowlisting two intentional lines with reasons. 14 MEDIUM: fixture URLs in
  `scripts/probe_writer_thinking.py` test data, never fetched.
- Mutation check, end of Phase D: 9/9 killed after one fix
  (`deferred-begin` survived; new read-then-write `atomic()` tests).
