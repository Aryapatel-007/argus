"""Measure whether thinking can be constrained per-request on qwen3.5:9b via Ollama.

Sprint 1 measurement script for the thinking-budget question in CLAUDE.md.
Not production code — do not import from src/argus/.

Run: python scripts/probe_thinking.py
Requires `ollama serve` running locally.
"""

import json
import subprocess
from importlib.metadata import version
from pathlib import Path

from ollama import Client

MODEL = "qwen3.5:9b"
PROMPT = "In one sentence, what is a state machine?"
NS_PER_S = 1_000_000_000
RUNS_PER_VARIANT = 2

NOTES_DIR = Path(__file__).resolve().parent.parent / "notes"
RESULTS_PATH = NOTES_DIR / "sprint1_thinking_probe.json"

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}

VARIANTS = [
    ("A_baseline", {}),
    ("B_think_false", {"think": False}),
    ("C_think_low", {"think": "low"}),
    ("D_think_false_json", {"think": False, "format": ANSWER_SCHEMA}),
    ("E_think_true_control", {"think": True}),
]


def print_versions() -> None:
    ollama_pkg_version = version("ollama")
    cli = subprocess.run(["ollama", "--version"], capture_output=True, text=True, check=False)
    cli_version = cli.stdout.strip() or cli.stderr.strip()
    print(f"ollama python package version: {ollama_pkg_version}")
    print(f"ollama CLI: {cli_version}")
    print()


def call(client: Client, extra_kwargs: dict):
    return client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        keep_alive="10m",
        options={"temperature": 0},
        **extra_kwargs,
    )


def record(variant: str, run_number: int, response=None, error: Exception | None = None) -> dict:
    if error is not None:
        return {
            "variant": variant,
            "run": run_number,
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
            "eval_count": None,
            "prompt_eval_count": None,
            "total_duration": None,
            "eval_duration": None,
            "load_duration": None,
            "tok_per_sec": None,
            "has_thinking_attr": None,
            "thinking_len": None,
            "content_len": None,
            "content": None,
        }

    message = response.message
    has_thinking_attr = hasattr(message, "thinking")
    if has_thinking_attr:
        thinking_len = len(message.thinking or "")
    else:
        thinking_len = None

    content = message.content or ""
    eval_count = response.eval_count
    eval_duration = response.eval_duration
    tok_per_sec = (
        eval_count / (eval_duration / NS_PER_S)
        if eval_count and eval_duration
        else None
    )

    return {
        "variant": variant,
        "run": run_number,
        "ok": True,
        "error": None,
        "eval_count": eval_count,
        "prompt_eval_count": response.prompt_eval_count,
        "total_duration": response.total_duration,
        "eval_duration": eval_duration,
        "load_duration": response.load_duration,
        "tok_per_sec": tok_per_sec,
        "has_thinking_attr": has_thinking_attr,
        "thinking_len": thinking_len,
        "content_len": len(content),
        "content": content[:200],
    }


def print_table(results: list[dict]) -> None:
    header = (
        f"{'variant':<22} {'run':<4} {'ok':<6} {'eval_cnt':<9} {'prompt_cnt':<11} "
        f"{'total_s':<9} {'eval_s':<9} {'load_s':<8} {'tok/s':<8} "
        f"{'thinking':<10} {'content':<9} content_preview / error"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        if not r["ok"]:
            print(
                f"{r['variant']:<22} {r['run']:<4} {'FAIL':<6} {'':<9} {'':<11} "
                f"{'':<9} {'':<9} {'':<8} {'':<8} {'':<10} {'':<9} {r['error']}"
            )
            continue

        total_s = f"{r['total_duration'] / NS_PER_S:.2f}"
        eval_s = f"{r['eval_duration'] / NS_PER_S:.2f}" if r["eval_duration"] else "n/a"
        load_s = f"{r['load_duration'] / NS_PER_S:.2f}" if r["load_duration"] else "n/a"
        tok_s = f"{r['tok_per_sec']:.2f}" if r["tok_per_sec"] is not None else "n/a"
        thinking = str(r["thinking_len"]) if r["has_thinking_attr"] else "NO ATTR"

        print(
            f"{r['variant']:<22} {r['run']:<4} {'OK':<6} {r['eval_count']:<9} "
            f"{r['prompt_eval_count']:<11} {total_s:<9} {eval_s:<9} {load_s:<8} "
            f"{tok_s:<8} {thinking:<10} {r['content_len']:<9} {r['content']}"
        )


def main() -> None:
    print_versions()

    client = Client()

    # Warm the model, discard the result.
    call(client, {})

    results = []
    for variant, extra_kwargs in VARIANTS:
        for run_number in range(1, RUNS_PER_VARIANT + 1):
            try:
                response = call(client, extra_kwargs)
                results.append(record(variant, run_number, response=response))
            except Exception as e:
                results.append(record(variant, run_number, error=e))

    print_table(results)

    NOTES_DIR.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
