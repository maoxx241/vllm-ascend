# AI Foundation Knowledge (Topic-Centered)

Single-source topic files with dual indexes for deployment/development retrieval.

## Layout

- `topics/`: one topic per file (`Core`, `Foundation`, `Deployment View`, `Development View`, `Details/Edge Cases`).
- `indexes/topic-index.json`: canonical topic metadata index.
- `indexes/term-alias-index.json`: alias -> canonical term/topic mapping.
- `indexes/view-index.json`: intent -> section routing index.
- `indexes/rule-index.json`: combo and model-compat rules.
- `model-profiles/`: model capability profiles.

## Guardrails

- `Core` is single source of truth for facts.
- Deployment/Development sections must not rewrite core facts.
- Evidence and conflict statuses come from generated KB entries.

Back to [shared index](../INDEX.md).
