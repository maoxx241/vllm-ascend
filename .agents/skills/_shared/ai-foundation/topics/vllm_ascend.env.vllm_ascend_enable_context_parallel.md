---
topic_id: vllm_ascend.env.vllm_ascend_enable_context_parallel
canonical_term: VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL
topic_kind: parameter
---

# VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL

## Core

- topic_id: `vllm_ascend.env.vllm_ascend_enable_context_parallel`
- canonical_term: `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `context_parallel`
- status/confidence: `aligned` / `0.98`
- source: `multi_source` / source_tags: code_definition, code_reference, docs_export
- semantics: 控制 Ascend 侧 Context Parallel 开关。
- aliases: `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL`, `vllm_ascend_enable_context_parallel`, `vllm-ascend-enable-context-parallel`, `vllm ascend enable context parallel`, `context_parallel`, `context parallel`, `context-parallel`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `context_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `numeric`
- accepted_values: int value
- constraints: 低卡数下收益低且配置复杂
- combo_effects: N/A

## Development View

- definition_ref: docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:161, docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:227, docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:321
- read_ref: vllm-ascend/vllm_ascend/envs.py:103, vllm-ascend/vllm_ascend/envs.py:103, vllm-ascend/vllm_ascend/utils.py:753
- effect_ref: vllm-ascend/vllm_ascend/utils.py:753
- web_refs: 6

## Details/Edge Cases

- failure_modes: KV 传输配置错误; 时延反而变高
- value_failure_signals: KV 传输配置错误; 时延反而变高
- recommendation: 优先在高并发长上下文场景启用并做 A/B。
- updated_at: 2026-03-11
