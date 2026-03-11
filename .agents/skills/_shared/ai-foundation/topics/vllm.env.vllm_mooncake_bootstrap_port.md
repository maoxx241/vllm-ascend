---
topic_id: vllm.env.vllm_mooncake_bootstrap_port
canonical_term: VLLM_MOONCAKE_BOOTSTRAP_PORT
topic_kind: parameter
---

# VLLM_MOONCAKE_BOOTSTRAP_PORT

## Core

- topic_id: `vllm.env.vllm_mooncake_bootstrap_port`
- canonical_term: `VLLM_MOONCAKE_BOOTSTRAP_PORT`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `network_serving`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: 控制服务监听、路由和 API 暴露。
- aliases: `VLLM_MOONCAKE_BOOTSTRAP_PORT`, `vllm_mooncake_bootstrap_port`, `vllm-mooncake-bootstrap-port`, `vllm mooncake bootstrap port`, `network_serving`, `network serving`, `network-serving`

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

- definition_ref: vllm/envs.py:1248
- read_ref: vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:1263, vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:1298, vllm/vllm/envs.py:174
- effect_ref: vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:1263, vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:1298, vllm/vllm/envs.py:174
- web_refs: 2

## Details/Edge Cases

- failure_modes: Address already in use; 健康检查 5xx
- value_failure_signals: Address already in use; 健康检查 5xx
- recommendation: 固定 host/port 并配套探活。
- updated_at: 2026-03-11
