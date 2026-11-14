Decide whether two facts about "$target_name" describe the SAME thing
(the same field of the same entity), even if the values differ.

# Known fact

entity_key: $known_key
content: $known_content

# New fact

entity_key: $new_key
content: $new_content

# Output

Return ONLY a JSON object: {"same_entity": true | false, "confidence": 0.0}.
