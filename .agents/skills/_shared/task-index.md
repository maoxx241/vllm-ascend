# Task Index (L0)

This task index maps natural-language goals to deterministic task chains.

| Task type | Subtasks | Read order | Skill chain |
| --- | --- | --- | --- |
| Single-node deployment | normalize intent -> choose profile -> render package -> start -> validate | [feature dictionary](deployment-config/concepts/feature-semantic-dictionary.md) -> [deployment-playbook](deployment-config/procedures/deployment-playbook.md) | `E2 deployment-assistant` |
| Environment bootstrap | inspect shell env -> install deps -> sanity checks -> summarize fixes | [env-bootstrap-baseline](ascend-foundation/procedures/env-bootstrap-baseline.md) | `A1 env-bootstrap` |
| Feature tuning request | normalize feature aliases -> map to CLI/env vars -> conflict check | [feature dictionary](deployment-config/concepts/feature-semantic-dictionary.md) | `E2 deployment-assistant` |
| Global parameter explanation | normalize term -> map to primary feature -> return usage and combo hints | [global parameter feature map](deployment-config/references/global-parameter-feature-map.md) -> [global combination guide](deployment-config/references/global-parameter-combination-guide.md) | `E2 deployment-assistant` |
| Parameter evidence verification | locate code refs -> check web refs -> mark confidence and conflict | [verification report](deployment-config/references/global-parameter-verification-report.md) -> [global parameter feature map](deployment-config/references/global-parameter-feature-map.md) | `E2 deployment-assistant` |
| Feature compatibility request | normalize features -> check model-feature matrix -> block unsupported | [compatibility matrix](vllm-ascend-core/concepts/model-feature-compatibility-matrix.md) -> [unsupported cases](troubleshooting/procedures/unsupported-feature-cases.md) | `E2 deployment-assistant` |
| Ambiguous phrase | return <=3 candidates -> ask single clarification -> continue | [feature dictionary](deployment-config/concepts/feature-semantic-dictionary.md) | `E2 deployment-assistant` |

## Default Profiles

- Primary: `qwen3-32b-w8a8`
- Backup: `qwen3-next-80b-a3b-instruct-w8a8`

Back to [INDEX.md](INDEX.md).
