# Writer probe: does `think=False` degrade digest prose?

Closes the Sprint 1/2 open item "`think=False` quality on writer-shaped
prompts". Script: `scripts/probe_writer_thinking.py` (plus
`scripts/_probe_writer_resume.py`, a throwaway that finished an interrupted
run). Raw data: `notes/writer-probe-raw.json`. Written up in Phase C,
2026-11-14, from the raw file only. No new generations were run.

## Setup

- Model `qwen3.5:9b` from config, temperature 0, no `format=` schema (free
  prose), 300s client-side kill as the runaway threshold.
- Three fact sets shaped like `RecalledFact` plus a change status:
  A = all unchanged, B = two modified + one added + two unchanged,
  C = ambiguous and thin (a regional caveat that partly contradicts a headline,
  a vague "mentioned in passing" fact, a bare date).
- 3 runs per set per think mode.

## Results

| Set | think=False | think=True |
|---|---|---|
| A no changes | 3/3 ok, 106 tokens, 11-16s | 1/3 ok (3415 tokens, 244s), 2 killed at 300s |
| B genuine changes | 3/3 ok, 124 tokens, 8-11s | 1/3 ok (4814 tokens, 218s), 2 killed at 300s |
| C ambiguous thin | 3/3 ok, 91 tokens, 4.6s | 0/1 ok, killed at 300s (runs 2-3 never finished) |

think=True: **5 of 7 generations exceeded 300s**. The two that finished took
218-244s for a paragraph the same model writes in ~10s without thinking.

think=False output was byte-identical across all three runs of each set
(temperature 0).

## Quality of the think=False digests (read by hand)

- **A:** every fact stated, all correctly framed as unchanged, no invented
  change. Equivalent to the one think=True answer.
- **B:** both modifications reported with the new value and the fact of a
  change ("replacing the previous figure for 2026", "whereas only March and
  July were previously noted"); the added scholarship called new; unchanged
  facts called unchanged. The think=True answer is no better: it drops the
  prior-value framing entirely.
- **C:** handles the contradiction correctly (3 years headline, 2 years for
  non-regional). Two soft faithfulness slips: "changes to the points test
  occurred" overstates a fact that only says changes were "mentioned in
  passing", and "These updates were announced on 3 August 2026" ties a bare,
  unchanged date fact to the new 485 facts, which no fact states.

## Verdict

`think=False` does not measurably degrade the writer. The writer stays on
`think=False` like every other role. The failure mode that does exist is
soft over-linking on thin evidence (set C), which thinking would not fix even
if it finished. The fix belongs in the graph: a deterministic faithfulness
check on `summarize` output (every date and number in the digest must appear
in an input fact) plus the eval harness's faithfulness score.

Thinking stays off. A 300s-plus step that sometimes never finishes cannot sit
inside a scheduled unattended run.
