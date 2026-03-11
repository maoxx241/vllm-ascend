# Task Index (L0)

This task index maps natural-language goals to deterministic task chains.

| Task type | Subtasks | Read order | Skill chain |
| --- | --- | --- | --- |
| Single-node deployment | normalize intent -> choose profile -> render package -> start -> validate | [feature dictionary](deployment-config/concepts/feature-semantic-dictionary.md) -> [deployment-playbook](deployment-config/procedures/deployment-playbook.md) | `E2 deployment-assistant` |
| Environment bootstrap | inspect shell env -> install deps -> sanity checks -> summarize fixes | [env-bootstrap-baseline](ascend-foundation/procedures/env-bootstrap-baseline.md) | `A1 env-bootstrap` |
| Feature tuning request | normalize feature aliases -> map to canonical topic -> conflict check | [ai-foundation index](ai-foundation/INDEX.md) -> [term alias index](ai-foundation/indexes/term-alias-index.json) | `E2 deployment-assistant` |
| Global parameter explanation | normalize term -> map to topic_id -> return deployment/dev/detail views | [topic index](ai-foundation/indexes/topic-index.json) -> [view index](ai-foundation/indexes/view-index.json) | `E2 deployment-assistant` |
| Parameter evidence verification | locate code refs -> check web refs -> mark confidence and conflict | [verification report](deployment-config/references/global-parameter-verification-report.md) -> [global parameter feature map](deployment-config/references/global-parameter-feature-map.md) | `E2 deployment-assistant` |
| Feature compatibility request | normalize features -> check model profile + rule index -> block unsupported | [model profiles](ai-foundation/model-profiles/qwen3-32b-w8a8.json) -> [rule index](ai-foundation/indexes/rule-index.json) | `E2 deployment-assistant` |
| Design analysis | classify design surface -> load imported manifest -> use design-analysis index -> map to atomic design skill | [design-analysis index](knowledge-governance/generated/design_analysis_index.json) -> [task-skill index](knowledge-governance/generated/task_skill_index.json) -> [code-knowledge-map](code-knowledge-map.md) | `E1 vllm-ascend-developer-assistant` |
| Model adaptation | classify model/API/runtime delta -> inspect imported manifest -> route to adaptation and precision skills | [imported knowledge manifest](knowledge-governance/generated/imported_knowledge_manifest.json) -> [task-skill index](knowledge-governance/generated/task_skill_index.json) | `E1 vllm-ascend-developer-assistant -> C1 model-adapter` |
| Debugging | classify crash/log/behavior issue -> inspect task-skill index -> pull code refs and gaps | [task-skill index](knowledge-governance/generated/task_skill_index.json) -> [imported knowledge report](knowledge-governance/generated/imported_knowledge_report.json) | `E1 vllm-ascend-developer-assistant -> C3 vllm-ascend-debug-assistant` |
| Upstream sync analysis | classify upstream delta -> inspect imported manifest -> group affected APIs/features -> scope follow-up | [imported knowledge report](knowledge-governance/generated/imported_knowledge_report.json) -> [task-skill index](knowledge-governance/generated/task_skill_index.json) | `E1 vllm-ascend-developer-assistant -> C2 sync-coordinator` |
| Release analysis | classify user-facing deltas -> inspect imported manifest by category/task -> compose release buckets | [imported knowledge report](knowledge-governance/generated/imported_knowledge_report.json) -> [task-skill index](knowledge-governance/generated/task_skill_index.json) | `E1 vllm-ascend-developer-assistant -> C4 release-assistant` |
| Operator development | inspect operator facts -> map to op/perf/precision skills -> check code paths and gaps | [imported knowledge manifest](knowledge-governance/generated/imported_knowledge_manifest.json) -> [design-analysis index](knowledge-governance/generated/design_analysis_index.json) | `E1 vllm-ascend-developer-assistant -> C5 op-developer` |
| Performance analysis | classify perf bottleneck -> map to perf/design knowledge -> inspect supporting code refs | [imported knowledge report](knowledge-governance/generated/imported_knowledge_report.json) -> [design-analysis index](knowledge-governance/generated/design_analysis_index.json) | `E1 vllm-ascend-developer-assistant -> C6 perf-assistant` |
| Knowledge maintenance / KB evolution | inspect execution state -> verify source/provenance -> rebuild generated indexes -> rerun scenario coverage | [verification handoff](knowledge-governance/provenance/verification_handoff.md) -> [execution state](knowledge-governance/provenance/execution_state.json) -> [domain index](knowledge-governance/generated/domain_index.json) | `E1 vllm-ascend-developer-assistant -> A18 knowledge-index-maintainer` |
| Ambiguous phrase | return <=3 candidates -> ask single clarification -> continue | [feature dictionary](deployment-config/concepts/feature-semantic-dictionary.md) | `E2 deployment-assistant` |

## Default Profiles

- Primary: `qwen3-32b-w8a8`
- Backup: `qwen3-next-80b-a3b-instruct-w8a8`
- Scenario regression: [skill-scenario coverage](knowledge-governance/generated/skill_scenario_coverage.json)

Back to [INDEX.md](INDEX.md).
