---
topic_id: vllm.env.s3_endpoint_url
canonical_term: S3_ENDPOINT_URL
topic_kind: parameter
---

# S3_ENDPOINT_URL

## Core

- topic_id: `vllm.env.s3_endpoint_url`
- canonical_term: `S3_ENDPOINT_URL`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `network_serving`
- status/confidence: `aligned` / `0.91`
- semantics: 控制服务监听、路由和 API 暴露。
- aliases: `S3_ENDPOINT_URL`, `s3_endpoint_url`, `s3-endpoint-url`, `s3 endpoint url`, `network_serving`, `network serving`, `network-serving`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `network_serving` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `numeric`
- accepted_values: int value
- constraints: 端口冲突会直接启动失败
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:623
- read_ref: vllm/vllm/envs.py:31, vllm/vllm/envs.py:623, vllm/vllm/envs.py:623
- effect_ref: vllm/vllm/envs.py:31, vllm/vllm/envs.py:623, vllm/vllm/envs.py:623
- web_refs: 2

## Details/Edge Cases

- failure_modes: Address already in use; 健康检查 5xx
- value_failure_signals: Address already in use; 健康检查 5xx
- recommendation: 固定 host/port 并配套探活。
- updated_at: 2026-03-05
