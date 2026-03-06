---
topic_id: vllm_ascend.env.vllm_ascend_enable_mlapo
canonical_term: VLLM_ASCEND_ENABLE_MLAPO
topic_kind: parameter
---

# VLLM_ASCEND_ENABLE_MLAPO

## Core

- topic_id: `vllm_ascend.env.vllm_ascend_enable_mlapo`
- canonical_term: `VLLM_ASCEND_ENABLE_MLAPO`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `aligned` / `0.95`
- source: `multi_source` / source_tags: code_definition, code_reference, docs_export
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `VLLM_ASCEND_ENABLE_MLAPO`, `vllm_ascend_enable_mlapo`, `vllm-ascend-enable-mlapo`, `vllm ascend enable mlapo`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `numeric`
- accepted_values: int value
- constraints: 错误组合可能影响稳定性
- combo_effects: N/A

## Development View

- definition_ref: docs/source/tutorials/models/DeepSeek-V3.2.md:141, docs/source/tutorials/models/DeepSeek-V3.2.md:200, docs/source/tutorials/models/DeepSeek-V3.2.md:249
- read_ref: vllm-ascend/vllm_ascend/attention/sfa_v1.py:400, vllm-ascend/vllm_ascend/attention/utils.py:329, vllm-ascend/vllm_ascend/envs.py:96
- effect_ref: vllm-ascend/vllm_ascend/attention/utils.py:329
- web_refs: 5

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-06
