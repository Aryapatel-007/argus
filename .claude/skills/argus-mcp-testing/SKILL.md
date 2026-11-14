---
name: argus-mcp-testing
description: Testing Argus's MCP servers (memory, browser) and the orchestrator's MCP client - in-memory Client tests, real stdio subprocess tests, stdout discipline, typed errors, idempotency across a killed caller, and Windows orphan-process checks. Use when adding or changing an MCP tool, server, or the client wrapper.
effort: high
---

# Testing Argus MCP servers

Written in-repo in Phase D: Anthropic's mcp-builder evaluation needs a paid
API key and teaches MCP SDK v1; the community tester only probes by hand.

## Three layers, each needed

1. **Backend, no MCP.** Call `MemoryServer` / browser backend methods
   directly. Fast, covers logic. Fake embeddings with a deterministic unit
   vector (see `tests/test_audit_fixes.py::fake_embed`).
2. **In-memory MCP.** `async with Client(build_server(db, run_id)) as c:`
   (MCP SDK v2). Covers tool registration, argument schemas, structured
   output, and how errors surface (`is_error=True` plus text).
3. **Real stdio subprocess.** `Client(stdio_client(StdioServerParameters(
   command=sys.executable, args=["-m", "argus.servers.memory", ...])))`.
   Covers spawn, `--run-id` validation, stdout discipline under real framing,
   clean shutdown, and process lifetime.

## Must-have tests for every server

- A tool call that fails surfaces as an error result, never a crashed server.
- Nothing is written to stdout except protocol frames (spawn it, send junk
  logging through every code path, the client must still parse).
- Startup refuses an unmigrated database and a nonexistent run id.
- Idempotency: calling a write tool twice with the same idempotency key
  returns the first result and writes nothing new.

## Windows process lifetime

Children are NOT killed with their parent. After the client exits, and after
the parent is hard-killed, list processes (PowerShell
`Get-CimInstance Win32_Process` filtered on the server's command line, or
`tasklist`) and assert no server remains. The MCP v2 stdio client uses a Job
Object on Windows; verify it, never assume stdin EOF ends the child.
