---
topic_id: vllm_ascend.env.openai_api_key
canonical_term: OPENAI_API_KEY
topic_kind: parameter
---

# OPENAI_API_KEY

## Core

- topic_id: `vllm_ascend.env.openai_api_key`
- canonical_term: `OPENAI_API_KEY`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `network_serving`
- status/confidence: `upstream_delta` / `0.68`
- source: `code` / source_tags: code_reference
- semantics: 控制服务监听、路由和 API 暴露。
- aliases: `OPENAI_API_KEY`, `openai_api_key`, `openai-api-key`, `openai api key`, `network_serving`, `network serving`, `network-serving`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `network_serving` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 端口冲突会直接启动失败
- combo_effects: N/A

## Development View

- definition_ref: examples/disaggregated_encoder/disagg_epd_proxy.py:650
- read_ref: vllm/vllm/benchmarks/lib/endpoint_request_func.py:146, vllm/vllm/entrypoints/cli/openai.py:33, vllm-ascend/examples/disaggregated_encoder/disagg_epd_proxy.py:650
- effect_ref: vllm/vllm/benchmarks/lib/endpoint_request_func.py:146, vllm/vllm/entrypoints/cli/openai.py:33, vllm-ascend/examples/disaggregated_encoder/disagg_epd_proxy.py:650
- web_refs: 3

## Details/Edge Cases

- failure_modes: Address already in use; 健康检查 5xx
- value_failure_signals: Address already in use; 健康检查 5xx
- recommendation: 固定 host/port 并配套探活。
- updated_at: 2026-03-11
