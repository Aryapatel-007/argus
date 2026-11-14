You consolidate what Argus knows about "$target_name" into a few
higher-level summaries.

# Current facts

$items

# Rules

- 1 to 3 summaries, each grouping related facts under a short `theme`
  (e.g. "cost", "admission", "schedule").
- `content`: one or two sentences, only from the facts above. Copy numbers
  and dates exactly.
- `importance` 0.0 to 1.0.

# Output

Return ONLY a JSON object: {"summaries": [{"theme": "...", "content": "...", "importance": 0.0}]}.
