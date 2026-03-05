# vLLM-Ascend Shared Knowledge Index (L0)

This index is optimized for weak-reasoning models. Always resolve user language into:
1. intent
2. canonical features
3. execution playbook

## Fast Route

| User intent | First read | Then read | Recommended skill |
| --- | --- | --- | --- |
| "帮我部署模型" / deployment | [task-index.md](task-index.md) | [deployment-playbook](deployment-config/procedures/deployment-playbook.md) | `vllm-ascend-deployment-assistant` |
| "先装环境" / bootstrap env | [task-index.md](task-index.md) | [env-bootstrap-baseline](ascend-foundation/procedures/env-bootstrap-baseline.md) | `vllm-ascend-env-bootstrap` |
| "开图/开并行/量化" | [feature dictionary](deployment-config/concepts/feature-semantic-dictionary.md) | [deployment-playbook](deployment-config/procedures/deployment-playbook.md) | `vllm-ascend-deployment-assistant` |
| "这个参数/环境变量是干什么的" | [ai-foundation index](ai-foundation/INDEX.md) | [topic index](ai-foundation/indexes/topic-index.json) | `vllm-ascend-deployment-assistant` |
| "这个参数结论靠谱吗" / evidence check | [verification report](deployment-config/references/global-parameter-verification-report.md) | [global parameter feature map](deployment-config/references/global-parameter-feature-map.md) | `vllm-ascend-deployment-assistant` |
| "这个特性能不能开" / compatibility | [compatibility matrix](vllm-ascend-core/concepts/model-feature-compatibility-matrix.md) | [unsupported cases](troubleshooting/procedures/unsupported-feature-cases.md) | `vllm-ascend-deployment-assistant` |
| Runtime error / startup fail | [error-index.md](error-index.md) | [deployment-playbook](deployment-config/procedures/deployment-playbook.md) | `vllm-ascend-deployment-assistant` |

## Weak Model Guardrails

- Only make one decision per step.
- If the phrase is ambiguous, return at most 3 candidates and ask one clarification.
- Prefer canonical features defined in the dictionary, never invent feature names.

## Related L0 Indexes

- [error-index.md](error-index.md)
- [task-index.md](task-index.md)
- [code-knowledge-map.md](code-knowledge-map.md)
- [repo-full-knowledge-map](vllm-ascend-core/references/repo-full-knowledge-map.md)
- [ai-foundation](ai-foundation/INDEX.md)
