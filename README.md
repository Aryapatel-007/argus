# Argus

Local-first autonomous multi-agent monitoring system. It watches a list of
targets (companies, topics, job postings), checks them on a schedule,
remembers what it finds, detects changes, and proposes actions that always
wait for human approval.

## Hard constraints

1. Everything runs locally — Ollama with local quantized models only, no
   paid APIs or hosted inference at runtime.
2. Zero recurring cost.
3. All data stays on the machine — no telemetry, no external logging.
4. Nothing externally visible happens without human approval.

## Hardware

- Asus ROG Strix G16, i7-13650HX, 16GB RAM, RTX 4060 Laptop (8GB VRAM)
- Windows 11, PowerShell 7.6, Python 3.11.9

VRAM is scarce (~7.6GB usable) — see `CLAUDE.md` for measured model memory
footprints and open decisions.

## Running the smoke test

```powershell
.\.venv\Scripts\Activate.ps1
ollama serve   # in a separate terminal, if not already running
python scripts/smoke_test.py
```

Reads model tags from `config/targets.yaml`, sends one chat prompt and one
embedding call, and prints timing/token stats from Ollama's own response
metadata.

---

This is Sprint 1. See `CLAUDE.md` for full architecture and current state, and
`notes/` for sprint-by-sprint measurements.
