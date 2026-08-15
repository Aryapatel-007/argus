"""One successful Python call to a local model through Ollama.

Run: python scripts/smoke_test.py
Requires `ollama serve` running locally.
"""

import sys
from pathlib import Path

import yaml
from ollama import Client

NS_PER_S = 1_000_000_000

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "targets.yaml"
CHAT_PROMPT = "In one sentence, what does a task scheduler do?"
EMBED_TEXT = "Master of Artificial Intelligence intake dates"


def main() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    models = config["models"]
    ollama_cfg = config["ollama"]
    client = Client(host=ollama_cfg["host"])

    try:
        chat_response = client.chat(
            model=models["chat"],
            messages=[{"role": "user", "content": CHAT_PROMPT}],
            options={"num_ctx": ollama_cfg["num_ctx"]},
        )

        eval_count = chat_response.eval_count
        eval_duration = chat_response.eval_duration
        tokens_per_second = (
            eval_count / (eval_duration / NS_PER_S)
            if eval_count and eval_duration
            else None
        )

        print(f"--- chat model: {models['chat']} ---")
        print(f"reply: {chat_response.message.content}")
        print(
            "model load / swap cost: "
            f"{chat_response.load_duration / NS_PER_S:.2f}s"
        )
        print(
            f"prompt_eval_count: {chat_response.prompt_eval_count}, "
            f"prompt_eval_duration: {chat_response.prompt_eval_duration / NS_PER_S:.2f}s"
        )
        print(
            f"eval_count: {eval_count}, "
            f"eval_duration: {eval_duration / NS_PER_S:.2f}s"
        )
        print(
            f"tokens/sec: {tokens_per_second:.2f}"
            if tokens_per_second is not None
            else "tokens/sec: n/a"
        )
        print(f"total_duration: {chat_response.total_duration / NS_PER_S:.2f}s")

        embed_response = client.embed(model=models["embed"], input=EMBED_TEXT)
        vector_length = len(embed_response.embeddings[0])

        print(f"\n--- embed model: {models['embed']} ---")
        print(f"embedding vector length (sqlite-vec column width): {vector_length}")

    except ConnectionError:
        print(
            "\nCould not reach Ollama. Check that `ollama serve` is running "
            f"and reachable at {ollama_cfg['host']}."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
