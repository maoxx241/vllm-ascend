---
topic_id: vllm_ascend.env.vllm_dp_size
canonical_term: VLLM_DP_SIZE
topic_kind: parameter
---

# VLLM_DP_SIZE

## Core

- topic_id: `vllm_ascend.env.vllm_dp_size`
- canonical_term: `VLLM_DP_SIZE`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `data_parallel`
- status/confidence: `aligned` / `0.98`
- source: `code` / source_tags: code_reference
- semantics: 通过副本扩展吞吐能力，并依赖 DP 地址和 RPC 协调。
- aliases: `VLLM_DP_SIZE`, `vllm_dp_size`, `vllm-dp-size`, `vllm dp size`, `data_parallel`, `data parallel`, `data-parallel`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `data_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 默认 1。
- value_shape: `numeric`
- accepted_values: int >= 1
- constraints: 通过 env fallback 注入时，dense 模型离线 DP>1 会报错。; 需与 VLLM_DP_RANK / VLLM_DP_MASTER_* 协同配置。
- combo_effects: 与 VLLM_DP_RANK、VLLM_DP_RANK_LOCAL、VLLM_DP_MASTER_IP/PORT 联动形成完整 DP 拓扑。

## Development View

- definition_ref: examples/offline_data_parallel.py:122
- read_ref: vllm/vllm/config/parallel.py:582, vllm/vllm/envs.py:133, vllm/vllm/envs.py:1059
- effect_ref: vllm/vllm/config/parallel.py:582, vllm/vllm/envs.py:133, vllm/vllm/envs.py:1059
- web_refs: 5

## Details/Edge Cases

- failure_modes: RPC 连接失败; 请求分发不均衡
- value_failure_signals: ValueError: Offline data parallel mode is not supported/useful for dense models.; ValueError: data_parallel_rank ... must be in the range [0, VLLM_DP_SIZE)
- recommendation: 固定 DP 地址和端口后再迭代性能参数。
- updated_at: 2026-03-11
