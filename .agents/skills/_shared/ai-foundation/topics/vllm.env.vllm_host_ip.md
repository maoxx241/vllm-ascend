---
topic_id: vllm.env.vllm_host_ip
canonical_term: VLLM_HOST_IP
topic_kind: parameter
---

# VLLM_HOST_IP

## Core

- topic_id: `vllm.env.vllm_host_ip`
- canonical_term: `VLLM_HOST_IP`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `network_serving`
- status/confidence: `aligned` / `0.91`
- semantics: 控制服务监听、路由和 API 暴露。
- aliases: `VLLM_HOST_IP`, `vllm_host_ip`, `vllm-host-ip`, `vllm host ip`, `network_serving`, `network serving`, `network-serving`

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

- definition_ref: vllm/envs.py:530
- read_ref: vllm/vllm/envs.py:15, vllm/vllm/envs.py:530, vllm/vllm/envs.py:530
- effect_ref: vllm/vllm/utils/network_utils.py:35, vllm/vllm/v1/executor/ray_executor.py:315
- web_refs: 2

## Details/Edge Cases

- failure_modes: Address already in use; 健康检查 5xx
- value_failure_signals: Address already in use; 健康检查 5xx
- recommendation: 固定 host/port 并配套探活。
- updated_at: 2026-03-05
