---
topic_id: vllm.env.vllm_dp_master_port
canonical_term: VLLM_DP_MASTER_PORT
topic_kind: parameter
---

# VLLM_DP_MASTER_PORT

## Core

- topic_id: `vllm.env.vllm_dp_master_port`
- canonical_term: `VLLM_DP_MASTER_PORT`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `data_parallel`
- status/confidence: `needs_manual_review` / `0.86`
- source: `code` / source_tags: code_definition
- semantics: 通过副本扩展吞吐能力，并依赖 DP 地址和 RPC 协调。
- aliases: `VLLM_DP_MASTER_PORT`, `vllm_dp_master_port`, `vllm-dp-master-port`, `vllm dp master port`, `data_parallel`, `data parallel`, `data-parallel`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `data_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 默认 0（由系统/逻辑进一步分配）。
- value_shape: `numeric_port`
- accepted_values: int >= 0 (建议有效监听端口)
- constraints: 端口需可用且与主节点地址匹配，冲突会导致初始化失败。
- combo_effects: 与 VLLM_DP_MASTER_IP、VLLM_DP_SIZE、VLLM_DP_RANK 联动。

## Development View

- definition_ref: vllm/envs.py:1051
- read_ref: vllm/vllm/config/parallel.py:586, vllm/vllm/envs.py:136, vllm/vllm/envs.py:1051
- effect_ref: vllm/vllm/utils/network_utils.py:159
- web_refs: 4

## Details/Edge Cases

- failure_modes: RPC 连接失败; 请求分发不均衡
- value_failure_signals: 端口占用/连接失败导致 DP 初始化错误。
- recommendation: 固定 DP 地址和端口后再迭代性能参数。
- updated_at: 2026-03-06
