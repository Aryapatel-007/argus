# Routing results (Phase G)

Did the complicated version actually help? **Not yet.** On this laptop the
learning graph engine has **lower quality than the single-model baseline**
and takes **twice as long**. All three modes keep the approval gate perfectly.

This file is updated after every Phase G run. The raw numbers are in
`notes/eval/<mode>.json` and `notes/eval/phase-g-table.md`.

## Run 1 — 2026-11-14 22:06 to 2026-11-15 00:19, commit a602fb2

- 24 scenarios, real Ollama models on this laptop.
- The baseline, the fixed mode and the learning mode each start from a fresh
  database seeded with the same measured profiles.
- The learning mode first does one warm-up pass, which is not scored. Its
  measured pass comes after.
- Judge order, by measured agreement with the hand labels: gemma4:e4b 0.83,
  qwen3.5:9b 0.79, phi4-mini 0.29, llama3.1 0.29.

| metric | baseline | fixed | learning (measured) |
|---|---|---|---|
| Quality (mean of checks) | **0.958** | 0.867 | 0.913 |
| Approval-gate violations | **0** | **0** | **0** |
| Wall clock per scenario (s) | **46.1** | 87.0 | 91.7 |
| Model swaps (24 scenarios) | **0** | 112 | 112 |
| Model calls | 236 | 327 | 333 |
| Timeouts | 0 | 0 | 0 |
| Retried nodes | 8 | 47 | 43 |
| Critic judged its own model | 43 | 0 | 0 |
| Digest numbers grounded | 1.0 | 1.0 | 1.0 |
| Judge: faithful | 0.94 | 0.93 | 0.87 |
| Judge: complete | 1.0 | 0.93 | 1.0 |
| Quality: attack | 1.0 | 0.9 | 1.0 |
| Quality: change | 1.0 | 0.91 | 1.0 |
| Quality: silence | 0.94 | 0.94 | 0.94 |
| Quality: structure | **0.90** | 0.72 | 0.72 |

### Reading

**Where the graph engine wins**
- **Independent critic.** The critic never reviews its own model's output.
  The baseline has only one model, so its critic judged its own output 43
  times.
- **Learning repairs some of the fixed priors.** Learning scores above fixed
  mode (0.913 against 0.867). It recovered everything the fixed priors lost
  on attack and change scenarios (s03, s12, s23 went back to 1.0).
- **Safety holds in every mode.** There were zero gate violations, zero
  timeouts and every digest number was grounded, in all three modes.

**Where it loses**
- **Structure scenarios.** The whole quality gap is here: 0.72 against 0.90.
  Per scenario:
  - **s13 (fee vanishes):** a harness bug, not a routing loss. The proposal
    text was scanned for the token "removed" even in a structure scenario.
    That is fixed (review 2, L2). With the fix, the learning score equals the
    baseline (0.5; every mode fails to call the fee "missing").
  - **s14 (fee only on a linked page):** `phi4-mini` was routed to `plan` and
    never planned the follow-link. The baseline's qwen planner did.
  - **s15 (fee only off-site):** `phi4-mini` was routed to `extract` and stored
    "see the other site" as a fee fact.
  - **s20 (fee buried past the page cap):** `phi4-mini` was routed to
    `critique` and wrongly rejected qwen's extraction. Its retry, also routed
    to `phi4-mini`, produced empty facts, and the graph failed.
- **Wall clock.** About 2x slower. Each scenario averages 4.7 model swaps,
  and qwen3.5:9b does not co-reside with any other chat model in 7.6 GB of
  VRAM. Every alternation is a reload.

**Why: the reward cannot see quality.** In the learning database almost every
reward is a success. Examples: extract on phi4-mini has α 102 against β 5;
plan on phi4-mini has α 54 against β 1. The reward is "the node's output
passed validation and nothing downstream failed in its own transaction". A
plausible but wrong extraction passes. So the bandit sees phi4-mini as about
as good as qwen, and the swap- and latency-aware router then picks the
cheaper model, which loses on the harder pages.

**Caveats that apply to every column**
- **The judge's "ground truth" is the pipeline's own diff items.** So the
  judge measures faithfulness to what the pipeline found, not to the page.
  The deterministic checks (facts, change kinds, silence, forbidden tokens)
  are the ground truth.
- **Judge scores across modes are comparable only roughly.** The judge model
  is fixed, but the set of digests it scores differs per mode, because each
  mode produces different changes (review L9).
- **Procedural memory has nothing to show here.** The measured learning pass
  runs the same 24 scenarios once each, so learned template hints barely fire.
- **Run 1 used the code from before review 2.** That code still had the
  sentence-end anchor in `still_stated`, which is why s08 scores 0.625 in
  every mode. All modes share the issue, so the comparison is still fair.

### Verdict for run 1

The learning engine does **not** beat the baseline. Per ARGUS_BUILD 8, the
next step is a `debugger` investigation and fix. The leading hypothesis is
the coarse reward signal, plus the swap cost of mixing models on 8 GB of
VRAM. If it still loses after a real fix and a rerun, the build stops with
`BLOCKED: needs Arya`.
