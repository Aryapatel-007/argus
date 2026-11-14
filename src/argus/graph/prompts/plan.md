You plan one monitoring check for Argus, a local agent that watches a target
and reports what changed.

# Target

- id: $target_id
- name: $target_name
- url: $target_url
- fields to find: $fields

# Templates you may use

$catalog

$hints_block

$trigger_block

# Rules

- Choose templates by name and fill their params. Do not invent templates.
- Use the target url exactly as given unless a template needs another url
  taken from the trigger above.
- `fields` is a list of the field names above, spelled exactly as given.
- `thought` is one sentence.

# Output

Return ONLY a JSON object: {"thought": "...", "steps": [{"template": "...", "params": {...}}]}.
