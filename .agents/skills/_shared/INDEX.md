# vLLM-Ascend Shared Knowledge Index (L0)

This index is optimized for weak-reasoning models. Always resolve user language into:
1. intent
2. canonical features
3. execution playbook

Top-level entry skills are restricted to exactly two:
- `vllm-ascend-deployment-assistant`
- `vllm-ascend-developer-assistant`

All other skills are internal Composer or Atomic skills and must not be exposed as first-hop user entry points here.

## Fast Route

| User intent | First read | Then read | Recommended skill |
| --- | --- | --- | --- |
| "帮我部署模型" / deployment | [task-index.md](task-index.md) | [deployment-playbook](deployment-config/procedures/deployment-playbook.md) | `vllm-ascend-deployment-assistant` |
| "先装环境" / bootstrap env | [task-index.md](task-index.md) | [env-bootstrap-baseline](ascend-foundation/procedures/env-bootstrap-baseline.md) | `vllm-ascend-deployment-assistant` |
| "开图/开并行/量化" | [feature dictionary](deployment-config/concepts/feature-semantic-dictionary.md) | [deployment-playbook](deployment-config/procedures/deployment-playbook.md) | `vllm-ascend-deployment-assistant` |
| "这个参数/环境变量是干什么的" | [ai-foundation index](ai-foundation/INDEX.md) | [topic index](ai-foundation/indexes/topic-index.json) | `vllm-ascend-deployment-assistant` |
| "这个参数结论靠谱吗" / evidence check | [verification report](deployment-config/references/global-parameter-verification-report.md) | [global parameter feature map](deployment-config/references/global-parameter-feature-map.md) | `vllm-ascend-deployment-assistant` |
| "这个特性能不能开" / compatibility | [compatibility matrix](vllm-ascend-core/concepts/model-feature-compatibility-matrix.md) | [unsupported cases](troubleshooting/procedures/unsupported-feature-cases.md) | `vllm-ascend-deployment-assistant` |
| "做研发分析/设计分析/架构分析" | [task-index.md](task-index.md) | [design-analysis index](knowledge-governance/generated/design_analysis_index.json) | `vllm-ascend-developer-assistant` |
| "模型适配/接口分析/接入新模型" | [task-index.md](task-index.md) | [imported knowledge manifest](knowledge-governance/generated/imported_knowledge_manifest.json) | `vllm-ascend-developer-assistant` |
| "调试日志/崩溃/行为异常" | [task-index.md](task-index.md) | [task-skill index](knowledge-governance/generated/task_skill_index.json) | `vllm-ascend-developer-assistant` |
| "profiling分析/性能瓶颈/回归归因" | [task-index.md](task-index.md) | [imported knowledge report](knowledge-governance/generated/imported_knowledge_report.json) | `vllm-ascend-developer-assistant` |
| "上游同步/接口影响评估" | [task-index.md](task-index.md) | [imported knowledge report](knowledge-governance/generated/imported_knowledge_report.json) | `vllm-ascend-developer-assistant` |
| "发布分析/变更归类" | [task-index.md](task-index.md) | [task-skill index](knowledge-governance/generated/task_skill_index.json) | `vllm-ascend-developer-assistant` |
| "知识维护/索引漂移/导入新知识" | [task-index.md](task-index.md) | [verification handoff](knowledge-governance/provenance/verification_handoff.md) | `vllm-ascend-developer-assistant` |
| Runtime error / startup fail | [error-index.md](error-index.md) | [deployment-playbook](deployment-config/procedures/deployment-playbook.md) | `vllm-ascend-deployment-assistant` |

## Weak Model Guardrails

- Only make one decision per step.
- If the phrase is ambiguous, return at most 3 candidates and ask one clarification.
- Prefer canonical features defined in the dictionary, never invent feature names.

## Related L0 Indexes

- [error-index.md](error-index.md)
- [task-index.md](task-index.md)
- [code-knowledge-map.md](code-knowledge-map.md)
- [skill-scenario coverage](knowledge-governance/generated/skill_scenario_coverage.json)
- [knowledge provenance](knowledge-governance/provenance/verification_handoff.md)
- [repo-full-knowledge-map](vllm-ascend-core/references/repo-full-knowledge-map.md)
- [imported knowledge manifest](knowledge-governance/generated/imported_knowledge_manifest.json)
- [design-analysis index](knowledge-governance/generated/design_analysis_index.json)
- [ai-foundation](ai-foundation/INDEX.md)
