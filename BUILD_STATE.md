# BUILD_STATE

Resume file for the ARGUS_BUILD.md build. Read this first, then CLAUDE.md,
then ARGUS_BUILD.md.

## Where we are

- **Phase:** C (audit and critical fixes) — next
- **Last completed:** Phase B, agent team
- **Test count:** 179 passing (baseline before Phase A: 179). Must never drop.
- **Blocked:** no

## Phase log

### Phase A — done (2026-11-14)

- Installed and pinned: playwright 1.65.2 (+ Chromium headless shell),
  apscheduler 3.11.3, fastapi 0.162.7, jinja2 3.1.6, pytest 9.2.0,
  pytest-asyncio 1.5.0, hypothesis 6.170.15. Already present and pinned:
  mcp 2.0.0, sqlite-vec 0.1.9, ollama 0.6.2, httpx 0.28.1, uvicorn 0.52.4.
- Rejected after reading the wheels: browser-use (D-002), deepeval (D-004),
  streamlit (D-003). Browser server will be raw Playwright.
- `pytest.ini` added; pytest and unittest both report 179.
- Versions, models, disk: `notes/environment.md`. Disk: C: 74.8 GB free.
- CLAUDE.md trimmed 285 -> 222 lines; Sprint 2 state moved to the end of
  notes/sprint-2.md. Install-up-front and decide-yourself rules recorded.
- ARGUS_BUILD.md saved into the repo so later sessions can re-read it.

### Phase B — done (2026-11-14)

- Frontmatter confirmed via claude-code-guide against the docs: `name`,
  `description` required; `model: opus`, `effort: medium|high|xhigh`,
  `tools`, `disallowedTools`. Unknown fields are silently ignored.
- 10 agents in `.claude/agents/`. Usage policy: D-007. Agents defined
  mid-session may not be listed until the next session starts; if an agent
  type is missing, run the same prompt through general-purpose and say so.

## Next steps

1. Phase C: auditor -> notes/audit.md; fix critical/high. Also write up the
   writer think probe (`notes/writer-probe-raw.json`, untracked until then).

## Backlog (medium/low, from audits)

(empty until Phase C)
