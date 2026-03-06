---
topic_id: vllm_ascend.env.vllm_dp_rank
canonical_term: VLLM_DP_RANK
topic_kind: parameter
---

# VLLM_DP_RANK

## Core

- topic_id: `vllm_ascend.env.vllm_dp_rank`
- canonical_term: `VLLM_DP_RANK`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `data_parallel`
- status/confidence: `upstream_delta` / `0.78`
- source: `code` / source_tags: code_reference
- semantics: 通过副本扩展吞吐能力，并依赖 DP 地址和 RPC 协调。
- aliases: `VLLM_DP_RANK`, `vllm_dp_rank`, `vllm-dp-rank`, `vllm dp rank`, `data_parallel`, `data parallel`, `data-parallel`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `data_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 默认 0。
- value_shape: `numeric`
- accepted_values: int >= 0
- constraints: 必须满足 0 <= rank < VLLM_DP_SIZE。
- combo_effects: 与 VLLM_DP_SIZE、VLLM_DP_RANK_LOCAL 一起决定本地/全局 DP 映射。

## Development View

- definition_ref: examples/offline_data_parallel.py:120
- read_ref: vllm/vllm/config/parallel.py:583, vllm/vllm/envs.py:131, vllm/vllm/envs.py:1040
- effect_ref: vllm/vllm/config/parallel.py:583, vllm/vllm/envs.py:131, vllm/vllm/envs.py:1040
- web_refs: 5

## Details/Edge Cases

- failure_modes: RPC 连接失败; 请求分发不均衡
- value_failure_signals: ValueError: data_parallel_rank ... must be in the range [0, data_parallel_size)
- recommendation: 固定 DP 地址和端口后再迭代性能参数。
- updated_at: 2026-03-06
