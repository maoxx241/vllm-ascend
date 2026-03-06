---
topic_id: vllm_ascend.env.ascend_transfer_timeout
canonical_term: ASCEND_TRANSFER_TIMEOUT
topic_kind: parameter
---

# ASCEND_TRANSFER_TIMEOUT

## Core

- topic_id: `vllm_ascend.env.ascend_transfer_timeout`
- canonical_term: `ASCEND_TRANSFER_TIMEOUT`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `upstream_delta` / `0.75`
- source: `multi_source` / source_tags: code_reference, docs_export
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `ASCEND_TRANSFER_TIMEOUT`, `ascend_transfer_timeout`, `ascend-transfer-timeout`, `ascend transfer timeout`, `general_runtime`, `general runtime`, `general-runtime`

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

- definition_ref: docs/source/user_guide/feature_guide/kv_pool.md:160, docs/source/user_guide/feature_guide/kv_pool.md:225, docs/source/user_guide/feature_guide/kv_pool.md:336
- read_ref: vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py:1097, vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:809, vllm-ascend/vllm_ascend/distributed/kv_transfer/utils/utils.py:52
- effect_ref: vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py:1097, vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:809, vllm-ascend/vllm_ascend/distributed/kv_transfer/utils/utils.py:52
- web_refs: 4

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-06
