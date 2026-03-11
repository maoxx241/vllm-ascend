---
topic_id: vllm.env.vllm_dp_rank_local
canonical_term: VLLM_DP_RANK_LOCAL
topic_kind: parameter
---

# VLLM_DP_RANK_LOCAL

## Core

- topic_id: `vllm.env.vllm_dp_rank_local`
- canonical_term: `VLLM_DP_RANK_LOCAL`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `data_parallel`
- status/confidence: `aligned` / `0.98`
- source: `code` / source_tags: code_definition
- semantics: 通过副本扩展吞吐能力，并依赖 DP 地址和 RPC 协调。
- aliases: `VLLM_DP_RANK_LOCAL`, `vllm_dp_rank_local`, `vllm-dp-rank-local`, `vllm dp rank local`, `data_parallel`, `data parallel`, `data-parallel`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `data_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 未设置时默认跟随 VLLM_DP_RANK。
- value_shape: `numeric_or_default`
- accepted_values: int >= -1
- constraints: 需与 VLLM_DP_SIZE、VLLM_DP_RANK 保持一致，避免本地拓扑错配。
- combo_effects: 与 VLLM_DP_RANK/VLLM_DP_SIZE 联动决定 local rank 视图。

## Development View

- definition_ref: vllm/envs.py:1055
- read_ref: vllm/vllm/config/parallel.py:584, vllm/vllm/envs.py:132, vllm/vllm/envs.py:1055
- effect_ref: vllm/vllm/config/parallel.py:584, vllm/vllm/envs.py:132, vllm/vllm/envs.py:1055
- web_refs: 4

## Details/Edge Cases

- failure_modes: RPC 连接失败; 请求分发不均衡
- value_failure_signals: 本地 rank 与实际拓扑不匹配时会导致通信/映射异常。
- recommendation: 固定 DP 地址和端口后再迭代性能参数。
- updated_at: 2026-03-11
