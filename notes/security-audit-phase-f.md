# Security audit, end of Phase F (2026-11-14)

Run with the `argus-security-audit` project skill: the scanner plus the
10-point checklist. Independent reviewer passes on the same code are listed
at the end; their findings are the most useful part of this audit.

## Step 1: scanner

`python scripts/security_scan.py` -> **0 blocking** (no unallowlisted
CRITICAL/HIGH). 18 findings total: 14 MEDIUM fixture URLs inside
`scripts/probe_writer_thinking.py` test data (never fetched), 2 MEDIUM
f-string SQL in the eval harness that only interpolates `?` placeholder lists
(allowlisted with reasons), 2 allowlisted lines with reasons (the thinking
probe's `think` variable, the POSIX-only `os.kill(pid, 0)`).

## Step 2: checklist

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Approval gate, structural | Holds, three layers | validator (`gate_bypasses`, hypothesis + negative controls), readiness (fails closed, property test + negative control), executor claim-before-act (D-023); mutants `validator-gate-check-off`, `readiness-ignores-unapproved-gates`, `executor-skips-approval-check` all killed |
| 2 | Side effects declared by code, not the model | Holds | legacy `tools.SIDE_EFFECT_TOOLS`; graph `NodeType.side_effect`; planner cannot create `notify` (not planner-visible) |
| 3 | Prompt injection from pages | Mitigated | every page-derived string fenced and escaped (`llm_nodes.fence`, spaced-tag variants); numbers must be on the page (extract), in the facts (summarize), in the digest (propose); off-site urls dropped from plans; eval scenarios s10, s11, s12, s22 attack this and score `forbidden` |
| 4 | Memory poisoning | Mitigated | facts reach memory only through a critique PASS; reflections are `summary:*` keys; `still_stated` keeps the page's own words, never a model rewording |
| 5 | Process control on Windows | Holds | no `os.kill` on Windows (scanner rule); MCP stdio children in a Job Object, verified by process-tree listing after hard kills (graph crash tests, browser kill test, scheduler kill test) |
| 6 | Network | Holds with one residual (below) | runtime traffic: Ollama on 127.0.0.1; the browser to target sites and same-site links. Browser blocks loopback/private/link-local origins other than the page's own, refuses cross-origin redirects, honours robots.txt. No telemetry SDK installed |
| 7 | SQL | Holds | parameters bound everywhere; f-strings only for placeholder lists or literal column lists (allowlisted) |
| 8 | MCP servers | Holds | no stdout writes (scanner rule + stdout discipline test); memory server refuses an unmigrated db and an unknown run id |
| 9 | Secrets | Holds | none in repo or config; child servers get an allowlisted environment (no stray tokens inherited) |
| 10 | Dashboard | Holds | 127.0.0.1 only; Host check on every page; CSRF token on every POST; Jinja autoescape verified with hostile strings; no docs/openapi routes; no external assets |

## Residual risks (accepted, documented)

- **DNS names that resolve to private addresses.** The browser's request
  filter judges IP literals and `localhost`; a public hostname that resolves
  to 10.x/192.168.x is not caught. Targets are configured by the owner, and
  redirects across origins are refused, so an attacker would need to control
  DNS for a watched site. Resolving every request before allowing it is the
  fix if targets ever come from untrusted input.
- **Model judgement under injection.** Fencing and validators make injected
  text inert in structure, but a local model can still be persuaded to
  mis-extract a value that IS on the page. The critic, cross-model checks for
  proposals, and above all the human approval are the backstops. Measured in
  Phase G by the attack scenarios.
- **`ARGUS_FAKE_EMBED=1`** in a user's shell would make the memory server
  write fake vectors; it logs a warning to stderr. Test-only switch.

## Independent reviewer passes on this code

Phase C fixes (2 high, 5 medium), Phase E (2 high, 8 medium), Phase E delta
(1 high, 3 medium), Phase F (1 high, 4 medium). Every finding was fixed with
a test; most have a mutant in `scripts/mutation_check.py` (22/22 killed).
Nothing critical is open.
