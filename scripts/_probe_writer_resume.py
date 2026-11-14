"""Resume-only: finish the 2 generations the original probe run was killed
mid-way through (C_ambiguous_thin / think=True / runs 2 and 3), merging into
the existing raw results file. Not a general-purpose script - throwaway.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import probe_writer_thinking as p  # noqa: E402
from argus import config  # noqa: E402
from argus.ollama_client import get_client  # noqa: E402

RAW = p.RAW_RESULTS_PATH
data = json.loads(RAW.read_text(encoding="utf-8"))

client = get_client(config.ollama_options().get("host"), p.RUNAWAY_TIMEOUT_S)
prompt = p.build_prompt("C_ambiguous_thin")

existing = data["C_ambiguous_thin"]["think_True"]
print(f"already have {len(existing)} think_True run(s) for C_ambiguous_thin")

start_run = len(existing) + 1
for run_number in range(start_run, p.RUNS_PER_VARIANT + 1):
    label = f"C_ambiguous_thin / think=True / run {run_number}"
    print(f"[{label}] starting...", flush=True)
    t0 = time.perf_counter()
    result = p.call(client, prompt, True)
    existing.append(result)
    RAW.write_text(json.dumps(data, indent=2), encoding="utf-8")

    if result["runaway"]:
        print(f"[{label}] RUNAWAY - killed after {result['elapsed_s']:.1f}s: {result['error']}", flush=True)
    elif not result["ok"]:
        print(f"[{label}] ERROR after {result['elapsed_s']:.1f}s: {result['error']}", flush=True)
    else:
        slow = " *** SLOW (>90s) ***" if result["elapsed_s"] > p.SLOW_WARN_S else ""
        print(
            f"[{label}] ok, {result['elapsed_s']:.1f}s eval_count={result['eval_count']} "
            f"thinking_chars={len(result['thinking'] or '')} "
            f"content_chars={len(result['content'] or '')}{slow}",
            flush=True,
        )

print("DONE")
