# Model profiles

Written by `scripts/profile_models.py` on 2026-11-14 13:00. Live numbers from this machine; the router reads them from `model_profiles`.

## Model-wide

| model | cold load s | warm load s | VRAM MB | GPU share | tok/s | think capable | think=False honoured | levels coerced |
|---|---|---|---|---|---|---|---|---|
| qwen3.5:9b | 3.93 | 0.15 | 5927 | 0.71 | 25.4 | 1 | 1 | 0 |
| phi4-mini:latest | 2.93 | 0.10 | 2753 | 1.00 | 80.9 | 0 | 1 | - |
| llama3.1:latest | 5.00 | 0.10 | 4955 | 1.00 | 51.1 | 0 | 1 | - |
| gemma4:e4b | 8.48 | 0.24 | 3149 | 0.32 | 33.1 | 1 | 1 | 0 |

## Per node type: quality / JSON valid rate / stability / mean latency s

| node type | qwen3.5:9b | phi4-mini:latest | llama3.1:latest | gemma4:e4b |
|---|---|---|---|---|
| critique | 1.00 / 1.00 / 1.00 / 2.2 | 1.00 / 1.00 / 1.00 / 1.4 | 1.00 / 1.00 / 1.00 / 1.5 | 1.00 / 1.00 / 1.00 / 2.0 |
| entity_match | 0.67 / 1.00 / 1.00 / 0.8 | 0.67 / 1.00 / 1.00 / 0.6 | 1.00 / 1.00 / 1.00 / 0.7 | 1.00 / 1.00 / 1.00 / 0.8 |
| extract | 1.00 / 1.00 / 1.00 / 6.8 | 1.00 / 1.00 / 1.00 / 4.5 | 1.00 / 1.00 / 1.00 / 3.6 | 1.00 / 1.00 / 1.00 / 3.6 |
| plan | 1.00 / 1.00 / 1.00 / 3.5 | 1.00 / 1.00 / 1.00 / 3.9 | 1.00 / 1.00 / 1.00 / 2.8 | 1.00 / 1.00 / 1.00 / 2.9 |
| propose_action | 1.00 / 1.00 / 1.00 / 5.5 | 1.00 / 1.00 / 0.00 / 3.5 | 1.00 / 1.00 / 1.00 / 3.2 | 1.00 / 1.00 / 1.00 / 3.5 |
| reflect | 1.00 / 1.00 / 1.00 / 5.3 | 1.00 / 1.00 / 1.00 / 4.5 | 1.00 / 1.00 / 0.00 / 3.6 | 1.00 / 1.00 / 1.00 / 4.0 |
| summarize | 1.00 / 1.00 / 1.00 / 2.7 | 1.00 / 1.00 / 0.00 / 2.1 | 1.00 / 1.00 / 1.00 / 1.9 | 1.00 / 1.00 / 1.00 / 2.1 |
| write_fact | 0.50 / 1.00 / 1.00 / 5.6 | 0.50 / 1.00 / 1.00 / 4.9 | 0.00 / 1.00 / 1.00 / 3.4 | 0.00 / 1.00 / 1.00 / 3.6 |

## Co-residency (both stayed loaded together)

- gemma4:e4b + llama3.1:latest: no
- gemma4:e4b + nomic-embed-text:latest: yes
- gemma4:e4b + phi4-mini:latest: yes
- gemma4:e4b + qwen3.5:9b: no
- llama3.1:latest + nomic-embed-text:latest: yes
- llama3.1:latest + phi4-mini:latest: no
- llama3.1:latest + qwen3.5:9b: no
- nomic-embed-text:latest + phi4-mini:latest: yes
- nomic-embed-text:latest + qwen3.5:9b: no
- phi4-mini:latest + qwen3.5:9b: no
