---
topic_id: vllm.env.vllm_enable_moe_dp_chunk
canonical_term: VLLM_ENABLE_MOE_DP_CHUNK
topic_kind: parameter
---

# VLLM_ENABLE_MOE_DP_CHUNK

## Core

- topic_id: `vllm.env.vllm_enable_moe_dp_chunk`
- canonical_term: `VLLM_ENABLE_MOE_DP_CHUNK`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `data_parallel`
- status/confidence: `aligned` / `0.91`
- semantics: 通过副本扩展吞吐能力，并依赖 DP 地址和 RPC 协调。
- aliases: `VLLM_ENABLE_MOE_DP_CHUNK`, `vllm_enable_moe_dp_chunk`, `vllm-enable-moe-dp-chunk`, `vllm enable moe dp chunk`, `data_parallel`, `data parallel`, `data-parallel`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `data_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `numeric`
- accepted_values: int value
- constraints: 错误地址/端口会导致调度与健康检查失败
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:1058
- read_ref: vllm/vllm/envs.py:138, vllm/vllm/envs.py:1058, vllm/vllm/envs.py:1059
- effect_ref: vllm/vllm/envs.py:138, vllm/vllm/envs.py:1058, vllm/vllm/envs.py:1059
- web_refs: 3

## Details/Edge Cases

- failure_modes: RPC 连接失败; 请求分发不均衡
- value_failure_signals: RPC 连接失败; 请求分发不均衡
- recommendation: 固定 DP 地址和端口后再迭代性能参数。
- updated_at: 2026-03-05
