# MCP Inspector session — memory server runbook

Manual checklist for running MCP Inspector against `argus.servers.memory`.
Not prose — follow top to bottom.

## 0. One-time: migrate the dev database to 002

`data/argus.db` exists but, as of this runbook being written, is migrated
only through `001c` — **not** `002`. `list_changes_since` needs `facts`,
`facts_vec`, and `approvals` to exist, or the server refuses to start
(`assert_migrated`, deliberately).

Check what's applied:

```powershell
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'src'); from argus.storage import db; print([r['filename'] for r in db.connect('data/argus.db').__enter__().execute('SELECT filename FROM schema_migrations')])"
```

If `002_facts_approvals.sql` is missing, apply it with the real migration
runner — this is additive only (new tables/columns, verified safe on a copy
of this exact file in Step 3) and is the orchestrator's sanctioned job, not
something to skip:

```powershell
$env:PYTHONPATH = "D:\Projects\argus\src"
.\.venv\Scripts\python.exe -m argus.storage.migrate
```

Confirms it worked: prints `applied 002_facts_approvals.sql` (or
`no pending migrations` if already applied).

## 1. Launch Inspector

Verified against the current MCP Inspector docs
(github.com/modelcontextprotocol/inspector), not assumed from v1:

```powershell
npx @modelcontextprotocol/inspector python src/argus/servers/memory.py -- --db-path data\argus.db --run-id inspector-session-1
```

Notes on this exact command:

- **No `--cli` / `--tui` flag** → launches the default interactive **web
  UI** (opens a browser tab). That's the mode this checklist assumes.
- **The direct-script form** (`python src/argus/servers/memory.py`), not
  `python -m argus.servers.memory` — the module already has a `sys.path`
  bootstrap for direct execution, so this needs no `PYTHONPATH` juggling on
  top of the `npx` invocation. (The `-m` form works too, but only with
  `PYTHONPATH` set first — see the repo-wide note below.)
- **The `--` separator is required**, not optional here: Inspector's docs
  are explicit that on the TUI *every* dash-argument meant for the server
  needs it, and the safe default is to always include it. Everything after
  `--` (`--db-path ...  --run-id ...`) goes to `memory.py`, not to Inspector.
- `--run-id inspector-session-1` is arbitrary — any string works. It only
  affects which `run_id` embedding calls get attributed to in `llm_calls`.

If this exact command has drifted from current Inspector behavior, run
`npx @modelcontextprotocol/inspector --help` to check first — do not assume
these flags are still current beyond whatever date this file was written.

## 2. Manual verification checklist

Work through these in order. Each is something to *look at*, not just glance
past.

- [ ] **Tool list shows all three**: `remember_fact`, `recall_facts`,
      `list_changes_since`. No more, no fewer.
- [ ] **Each tool's schema looks right** when expanded in the UI:
  - `remember_fact`: `target_id`, `entity_key`, `content` required;
    `attributes`, `source_url`, `confidence` optional.
  - `recall_facts`: `query` required; `target_id`, `limit`, `min_confidence`
    optional.
  - `list_changes_since`: `since` required; `target_id`, `kinds`, `limit`
    optional.
- [ ] **One real `remember_fact` call.** Example:
      `{"target_id": "inspector_test", "entity_key": "note", "content": "Inspector session smoke test."}`
      Should return `change_kind: "added"`, `version: 1`, `previous_fact_id: null`.
- [ ] **One real `recall_facts` call.** Example:
      `{"query": "inspector session"}`
      Should return the fact just written, with `score`, `relevance`,
      `recency` all populated as numbers (not null, not missing).
- [ ] **One real `list_changes_since` call.** Example:
      `{"since": "2020-01-01T00:00:00.000Z"}`
      Should return at least the fact just written, with `change_kind:
      "added"` and `previous_content: null`.
- [ ] **`structured_content` is non-null on all three calls above.** This is
      the exact bug class fixed in Step 6: a tool can register successfully,
      answer with `isError: false`, and still silently return a *structured*
      result of `null` if a return-type annotation is malformed. Inspector's
      response view shows this as a separate "Structured Content" panel from
      the plain-text content — check that panel specifically on each of the
      three calls, don't just check that the call didn't error.

## 3. What "working" looks like vs. what a real failure looks like

| Signal | Working | Real failure |
|---|---|---|
| Inspector connection | Connects, tool list populates within a couple seconds | Spinner that never resolves, or an immediate "failed to connect" — check the terminal Inspector was launched from for a Python traceback on stderr |
| Tool call result | Green/success state, a "Structured Content" panel with actual fields filled in | Red/error state, OR (the subtle one) success state but "Structured Content" showing `null`/empty — that's the quoted-annotation bug class, not a connection problem |
| `remember_fact` called twice, same content | Second call returns `change_kind: "unchanged"`, not a duplicate `"added"` | A second `"added"` means the change-detection logic broke |
| Server process | The terminal running the `npx` command shows nothing beyond one `[argus-memory] serving ...` line on stderr, ever | Any Python traceback in that terminal, or anything printed that looks like JSON-RPC frames leaking as visible text (would mean something wrote to stdout — see memory.py's module docstring on why that's fatal) |
| Closing Inspector | Ctrl-C in the terminal cleanly ends the `python` subprocess | Process hangs after Ctrl-C and needs a manual kill — worth noting if it happens, not expected |

If Inspector itself won't launch (the `npx` command fails before ever
reaching the tool list), that's an Inspector/Node problem, not a
`memory.py` problem — check `node --version` (need 22.19.0+, confirmed in
Step 0) and `npx --version` first.
