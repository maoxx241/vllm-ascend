---
topic_id: vllm.env.vllm_nixl_side_channel_host
canonical_term: VLLM_NIXL_SIDE_CHANNEL_HOST
topic_kind: parameter
---

# VLLM_NIXL_SIDE_CHANNEL_HOST

## Core

- topic_id: `vllm.env.vllm_nixl_side_channel_host`
- canonical_term: `VLLM_NIXL_SIDE_CHANNEL_HOST`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `network_serving`
- status/confidence: `aligned` / `0.91`
- semantics: 控制服务监听、路由和 API 暴露。
- aliases: `VLLM_NIXL_SIDE_CHANNEL_HOST`, `vllm_nixl_side_channel_host`, `vllm-nixl-side-channel-host`, `vllm nixl side channel host`, `network_serving`, `network serving`, `network-serving`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `network_serving` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 端口冲突会直接启动失败
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:1228
- read_ref: vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py:522, vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py:614, vllm/vllm/envs.py:171
- effect_ref: vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py:522, vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py:614, vllm/vllm/envs.py:171
- web_refs: 2

## Details/Edge Cases

- failure_modes: Address already in use; 健康检查 5xx
- value_failure_signals: Address already in use; 健康检查 5xx
- recommendation: 固定 host/port 并配套探活。
- updated_at: 2026-03-05
