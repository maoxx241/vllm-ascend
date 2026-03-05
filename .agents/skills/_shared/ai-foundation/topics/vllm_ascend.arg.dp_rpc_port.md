---
topic_id: vllm_ascend.arg.dp_rpc_port
canonical_term: --dp-rpc-port
topic_kind: parameter
---

# --dp-rpc-port

## Core

- topic_id: `vllm_ascend.arg.dp_rpc_port`
- canonical_term: `--dp-rpc-port`
- kind/scope: `arg` / `vllm_ascend`
- stage: `startup`
- primary_feature: `data_parallel`
- status/confidence: `needs_manual_review` / `0.86`
- semantics: 通过副本扩展吞吐能力，并依赖 DP 地址和 RPC 协调。
- aliases: `--dp-rpc-port`, `dp-rpc-port`, `dp_rpc_port`, `dp rpc port`, `dprpcport`, `data_parallel`, `data parallel`, `data-parallel`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `data_parallel` 查看稳定原理。

## Deployment View

- default_behavior: Port for data parallel master node.
- value_shape: `numeric`
- accepted_values: int value
- constraints: 错误地址/端口会导致调度与健康检查失败
- combo_effects: N/A

## Development View

- definition_ref: examples/external_online_dp/launch_online_dp.py:15
- read_ref: vllm-ascend/examples/external_online_dp/launch_online_dp.py:28, vllm-ascend/examples/external_online_dp/launch_online_dp.py:28, vllm-ascend/examples/external_online_dp/launch_online_dp.py:41
- effect_ref: vllm-ascend/examples/external_online_dp/launch_online_dp.py:28, vllm-ascend/examples/external_online_dp/launch_online_dp.py:28, vllm-ascend/examples/external_online_dp/launch_online_dp.py:41
- web_refs: 4

## Details/Edge Cases

- failure_modes: RPC 连接失败; 请求分发不均衡
- value_failure_signals: RPC 连接失败; 请求分发不均衡
- recommendation: 固定 DP 地址和端口后再迭代性能参数。
- updated_at: 2026-03-05
