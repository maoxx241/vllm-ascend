---
topic_id: vllm.env.vllm_moe_dp_chunk_size
canonical_term: VLLM_MOE_DP_CHUNK_SIZE
topic_kind: parameter
---

# VLLM_MOE_DP_CHUNK_SIZE

## Core

- topic_id: `vllm.env.vllm_moe_dp_chunk_size`
- canonical_term: `VLLM_MOE_DP_CHUNK_SIZE`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `data_parallel`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: 通过副本扩展吞吐能力，并依赖 DP 地址和 RPC 协调。
- aliases: `VLLM_MOE_DP_CHUNK_SIZE`, `vllm_moe_dp_chunk_size`, `vllm-moe-dp-chunk-size`, `vllm moe dp chunk size`, `data_parallel`, `data parallel`, `data-parallel`

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

- definition_ref: vllm/envs.py:1069
- read_ref: vllm/vllm/envs.py:138, vllm/vllm/envs.py:1065, vllm/vllm/envs.py:1067
- effect_ref: vllm/vllm/model_executor/layers/quantization/mxfp4.py:765
- web_refs: 3

## Details/Edge Cases

- failure_modes: RPC 连接失败; 请求分发不均衡
- value_failure_signals: RPC 连接失败; 请求分发不均衡
- recommendation: 固定 DP 地址和端口后再迭代性能参数。
- updated_at: 2026-03-11
