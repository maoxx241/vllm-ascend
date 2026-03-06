---
topic_id: vllm_ascend.env.vllm_use_v1
canonical_term: VLLM_USE_V1
topic_kind: parameter
---

# VLLM_USE_V1

## Core

- topic_id: `vllm_ascend.env.vllm_use_v1`
- canonical_term: `VLLM_USE_V1`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `upstream_delta` / `0.55`
- source: `multi_source` / source_tags: docs_export, tests_yaml
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `VLLM_USE_V1`, `vllm_use_v1`, `vllm-use-v1`, `vllm use v1`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 错误组合可能影响稳定性
- combo_effects: N/A

## Development View

- definition_ref: docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:159, docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:225, docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:92
- read_ref: N/A
- effect_ref: N/A
- web_refs: 4

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-06
