# Environment (Phase A, 2026-11-14)

Measured with live commands on this date. If a number here disagrees with a
live command later, the live command wins: re-run it and update this file.

## Machine

- Windows 11 Home 10.0.26200, PowerShell 7.6
- RTX 4060 Laptop, 8188 MiB total (nvidia-smi), driver 610.88
- Free disk: C: 74.8 GB (Ollama models and Playwright browsers live here),
  D: 308.5 GB (repo)

## Python

- Python 3.11.9 in `.venv\` at repo root. `python` resolves inside `.venv`
  after `.\.venv\Scripts\Activate.ps1`.
- Test runner: pytest 9.2.0 (`python -m pytest`). It also collects the older
  `unittest.TestCase` suites, so `python -m unittest discover -s tests` keeps
  working for the files written before Phase A.
- Baseline before Phase A changes: **179 tests passing** (both runners agree).

## Ollama

- CLI / server: 0.24.0. An update to 0.40.16 is offered by the tray app.
  Not taken: every measurement in notes/ was made on 0.24.0, and an engine
  update mid-build would invalidate the profiler's priors. Revisit only with a
  full profiler re-run.
- Python client package: 0.6.2 (different number, different thing).
- Local models (`ollama list`):

| Tag | Size on disk |
|---|---|
| qwen3.5:9b | 6.6 GB |
| phi4-mini:latest | 2.5 GB |
| nomic-embed-text:latest | 274 MB |
| llama3.1:latest | 4.9 GB |
| deepseek-r1:8b | 5.2 GB |
| gemma4:e4b | 9.6 GB |
| llama3.2-vision:latest | 7.8 GB |
| sunday:latest | 4.9 GB (custom, unknown base) |

## Python packages (direct deps, pinned in requirements*.txt)

| Package | Version | Used by |
|---|---|---|
| ollama | 0.6.2 | every model call |
| mcp | 2.0.0 | memory + browser servers, orchestrator client |
| sqlite-vec | 0.1.9 | facts_vec |
| playwright | 1.65.2 | browser server. Chromium headless shell 159.0.8154.6 installed via `python -m playwright install chromium` |
| apscheduler | 3.11.3 | scheduler |
| fastapi | 0.162.7 | dashboard |
| jinja2 | 3.1.6 | dashboard templates |
| uvicorn | 0.52.4 | dashboard server (already present via mcp) |
| httpx | 0.28.1 | async HTTP |
| pydantic | 2.13.4 | schemas |
| pyyaml | 6.0.3 | config |
| pytest | 9.2.0 | dev |
| pytest-asyncio | 1.5.0 | dev |
| hypothesis | 6.170.15 | dev, graph invariant property tests |

`pip check`: no broken requirements.

## Rejected at install time

- **browser-use 1.0.4.** Inspected the wheel before installing. Telemetry
  (posthog) is on unless `ANONYMIZED_TELEMETRY=false`, `update_check.py`
  calls PyPI at runtime, several code paths target `api.browser-use.com`,
  there is no native Ollama provider (only Anthropic/Google/OpenAI via
  pydantic-ai), and it pins `mcp==2.3.0` and `pydantic==2.14.3` exactly.
  Fallback taken: raw Playwright plus our own LLM extraction node. D-002.
- **deepeval 4.3.4.** Hard dependency on `posthog`, `confident-trace` and
  OpenTelemetry exporters, and requires `click<8.4.0` while the installed
  click is 8.4.2. Telemetry is opt-out, not opt-in. The eval harness is
  built in-repo instead (Phase F step 7). D-004.
- **streamlit.** Usage-stats collection on by default. D-003.

## One-time network use (install time only, never at runtime)

- `pip install` from PyPI.
- `python -m playwright install chromium` from the Playwright CDN.
- `ollama pull` for pool models (Phase E step 4).
