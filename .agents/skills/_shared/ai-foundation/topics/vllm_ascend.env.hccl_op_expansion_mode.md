---
topic_id: vllm_ascend.env.hccl_op_expansion_mode
canonical_term: HCCL_OP_EXPANSION_MODE
topic_kind: parameter
---

# HCCL_OP_EXPANSION_MODE

## Core

- topic_id: `vllm_ascend.env.hccl_op_expansion_mode`
- canonical_term: `HCCL_OP_EXPANSION_MODE`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `upstream_delta` / `0.75`
- source: `multi_source` / source_tags: code_reference, docs_export
- semantics: 控制 HCCL 算子展开策略（常见 AIV），影响通信兼容性与性能。
- aliases: `HCCL_OP_EXPANSION_MODE`, `hccl_op_expansion_mode`, `hccl-op-expansion-mode`, `hccl op expansion mode`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: 未设置时使用 HCCL 默认展开策略。
- value_shape: `enum`
- accepted_values: AIV, default(HCCL)
- constraints: 仅在 HCCL 版本与驱动组合支持时生效。
- combo_effects: 与 TP/DP/CP 并行拓扑强耦合，不匹配会影响通信性能。

## Development View

- definition_ref: docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md:176, docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:158, docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:224
- read_ref: vllm-ascend/vllm_ascend/utils.py:477, vllm-ascend/vllm_ascend/utils.py:525
- effect_ref: vllm-ascend/vllm_ascend/utils.py:477
- web_refs: 5

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: HCCL 初始化或通信超时告警。
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-11
