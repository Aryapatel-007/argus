# Sprint 0 — measurements

## GPU driver

- NVIDIA driver: 555.97 -> 610.88, CUDA 13.3.
- Before: Ollama GPU discovery failed entirely. CUDA and ROCm backends both
  timed out, fell back to CPU-only, `total_vram="0 B"`.
- After: `library=CUDA`, 7.6GB available.

## qwen3.5:9b memory footprint (Q4_K_M)

- ~8.2GB required against a ~7.6GB ceiling:
  - 4.0GB GPU weights
  - 2.2GB output layer
  - 1.4GB KV cache
  - 0.55GB compute graph
- Output layer pushed to CPU: one layer of 33, but disproportionately heavy,
  which is why `ollama ps` reports a ~70/30 GPU/CPU split.

## Throughput

- 24.3 tok/s in the Sprint 0 smoke test.
- ~19-21 tok/s observed later.
- Cold load: 6.00s. Warm load: 0.12s.

## Context window

- Reducing `num_ctx` from 4096 to 2048 did not meaningfully help. Bottleneck
  is weights, not KV cache.

## Embeddings

- `nomic-embed-text` output dimension: 768.

## Environment versions

Measured during the Sprint 1 thinking probe. Two different numbers, easy to
confuse:

- `ollama` Python package (PyPI client library): 0.6.2
- Ollama CLI / server binary: 0.24.0
