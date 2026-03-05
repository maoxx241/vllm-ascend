---
topic_id: vllm_ascend.env.vllm_dp_master_ip
canonical_term: VLLM_DP_MASTER_IP
topic_kind: parameter
---

# VLLM_DP_MASTER_IP

## Core

- topic_id: `vllm_ascend.env.vllm_dp_master_ip`
- canonical_term: `VLLM_DP_MASTER_IP`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `data_parallel`
- status/confidence: `aligned` / `0.98`
- semantics: 通过副本扩展吞吐能力，并依赖 DP 地址和 RPC 协调。
- aliases: `VLLM_DP_MASTER_IP`, `vllm_dp_master_ip`, `vllm-dp-master-ip`, `vllm dp master ip`, `data_parallel`, `data parallel`, `data-parallel`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `data_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 默认 127.0.0.1。
- value_shape: `string_ip`
- accepted_values: IPv4/hostname
- constraints: 多节点场景需保证地址可达且与端口一致。
- combo_effects: 与 VLLM_DP_MASTER_PORT、VLLM_DP_SIZE、VLLM_DP_RANK 联动。

## Development View

- definition_ref: examples/offline_data_parallel.py:123
- read_ref: vllm/vllm/config/parallel.py:585, vllm/vllm/envs.py:135, vllm/vllm/envs.py:1049
- effect_ref: vllm/vllm/config/parallel.py:585, vllm/vllm/envs.py:135, vllm/vllm/envs.py:1049
- web_refs: 5

## Details/Edge Cases

- failure_modes: RPC 连接失败; 请求分发不均衡
- value_failure_signals: DP 组初始化连接失败、RPC/Socket 无法建立。
- recommendation: 固定 DP 地址和端口后再迭代性能参数。
- updated_at: 2026-03-05
