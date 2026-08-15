"""One successful Python call to a local model through Ollama.

Run: python scripts/smoke_test.py
Requires `ollama serve` running locally.
"""

import time
from pathlib import Path

import yaml
from ollama import Client

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
        start = time.perf_counter()
        chat_response = client.chat(
            model=models["chat"],
            messages=[{"role": "user", "content": CHAT_PROMPT}],
            options={"num_ctx": ollama_cfg["num_ctx"]},
        )
        latency = time.perf_counter() - start

        print(f"--- chat model: {models['chat']} ---")
        print(f"reply: {chat_response.message.content}")
        print(f"latency: {latency:.2f}s")
        print(f"tokens generated (eval_count): {chat_response.eval_count}")

        embed_response = client.embed(model=models["embed"], input=EMBED_TEXT)
        vector_length = len(embed_response.embeddings[0])

        print(f"\n--- embed model: {models['embed']} ---")
        print(f"embedding vector length (sqlite-vec column width): {vector_length}")

    except ConnectionError:
        print(
            "\nCould not reach Ollama. Check that `ollama serve` is running "
            f"and reachable at {ollama_cfg['host']}."
        )


if __name__ == "__main__":
    main()
