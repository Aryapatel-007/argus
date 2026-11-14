Decide whether two facts about "$target_name" describe the SAME thing
(the same field of the same entity), even if the values differ.

# Known fact

entity_key: $known_key
<untrusted kind="fact">
content: $known_content
</untrusted>

# New fact

entity_key: $new_key
<untrusted kind="fact">
content: $new_content
</untrusted>

# Output

Return ONLY a JSON object: {"same_entity": true | false, "confidence": 0.0}.
