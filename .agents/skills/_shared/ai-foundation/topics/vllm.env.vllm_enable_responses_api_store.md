---
topic_id: vllm.env.vllm_enable_responses_api_store
canonical_term: VLLM_ENABLE_RESPONSES_API_STORE
topic_kind: parameter
---

# VLLM_ENABLE_RESPONSES_API_STORE

## Core

- topic_id: `vllm.env.vllm_enable_responses_api_store`
- canonical_term: `VLLM_ENABLE_RESPONSES_API_STORE`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `network_serving`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: 控制服务监听、路由和 API 暴露。
- aliases: `VLLM_ENABLE_RESPONSES_API_STORE`, `vllm_enable_responses_api_store`, `vllm-enable-responses-api-store`, `vllm enable responses api store`, `network_serving`, `network serving`, `network-serving`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `network_serving` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `numeric`
- accepted_values: int value
- constraints: 端口冲突会直接启动失败
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:1398
- read_ref: vllm/vllm/entrypoints/openai/responses/serving.py:238, vllm/vllm/entrypoints/openai/responses/serving.py:241, vllm/vllm/entrypoints/openai/responses/serving.py:331
- effect_ref: vllm/vllm/entrypoints/openai/responses/serving.py:238, vllm/vllm/entrypoints/openai/responses/serving.py:241, vllm/vllm/entrypoints/openai/responses/serving.py:331
- web_refs: 2

## Details/Edge Cases

- failure_modes: Address already in use; 健康检查 5xx
- value_failure_signals: Address already in use; 健康检查 5xx
- recommendation: 固定 host/port 并配套探活。
- updated_at: 2026-03-06
