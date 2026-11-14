---
name: argus-security-audit
description: Security audit of Argus agent code - hosted-model or telemetry leaks, approval-gate bypasses, prompt injection from fetched pages into tool arguments, unsafe subprocess and process-kill use on Windows, secrets, SQL built from strings, MCP stdout discipline. Use at the end of Phase D and Phase F, before any release, and whenever a new tool, node type, prompt or MCP server is added.
effort: high
---

# Argus security audit

Written in-repo in Phase D because no vettable external skill fit (see
`notes/skills.md`: Cloudflare's security-audit skill was rejected for broken
Windows validators and an open, unanswered report of it wiping a Windows user
profile). Everything here runs locally and sends nothing anywhere.

## Step 1: the scanner (mechanical tripwire)

```powershell
python scripts/security_scan.py
```

Exit 1 means a CRITICAL or HIGH finding that is not allowlisted. Allowlist
only with a reason on the line: `# security-scan: ok <reason>`. Never
allowlist hosted-llm-sdk, telemetry or secret-literal.

## Step 2: the checklist (what a regex cannot see)

Work through each item and cite file:line for every answer.

1. **Approval gate, structural.** Every `execute_action` node is unreachable
   from a root once `approval_gate` nodes are deleted
   (`graph/validator.gate_bypasses`). The scheduler only marks an action ready
   after its gate ancestors are `done`, and the executor re-reads the
   `approvals` row (`status = 'approved'`, matching idempotency key) at run
   time. Three independent checks; verify all three exist and are tested.
2. **Side effects declared by the tool, not the model.** Legacy loop:
   `tools.SIDE_EFFECT_TOOLS`. Graph: `NodeType.side_effect`. A planner flag
   may ADD externality, never remove it.
3. **Prompt injection from fetched pages.** Page text is untrusted input.
   - It must reach prompts only inside an `<untrusted>` fence, and the prompt
     must say instructions inside the fence are data.
   - Nothing from page text may choose a tool, a URL to fetch outside the
     target's domain, a file path, or an approval decision without passing a
     deterministic validator.
   - `extract` output is checked against the page (numbers must appear in
     the page text) before any critic sees it.
   - `propose_action` content is checked against the digest, and every
     proposal goes through cross_model agreement AND a human.
4. **Memory poisoning.** Facts written to memory come only through
   `remember_fact` after a critique PASS. Reflection summaries are marked as
   such (entity key prefix `summary:`) and never treated as observed facts.
5. **Process control on Windows.** No `os.kill` (it is TerminateProcess).
   Child servers are tracked and their absence after shutdown is checked by
   process listing, not assumed.
6. **Network.** Runtime network use is limited to: Ollama on 127.0.0.1, the
   browser fetching target URLs from `config/targets.yaml` (and same-domain
   links), and nothing else. No telemetry SDK is installed
   (`opentelemetry-api` arrives with mcp and is a no-op without an SDK; never
   install `opentelemetry-sdk`).
7. **SQL.** Parameters always bound. f-strings only for placeholder lists
   (`", ".join("?" ...)`), never for values.
8. **MCP servers.** No stdout writes (`print` without `file=sys.stderr`).
   Startup refuses an unmigrated database.
9. **Secrets.** None in the repo, none in config. Any future outbound action
   that needs a credential is a `BLOCKED: needs Arya` item.
10. **Dashboard.** Binds 127.0.0.1 only. State-changing routes are POST and
    check a per-process CSRF token.

## Output

Write findings to `notes/security-audit-<phase>.md`: severity
(critical/high/medium/low), file:line, failure scenario, fix. Fix every
critical before the phase commit.
