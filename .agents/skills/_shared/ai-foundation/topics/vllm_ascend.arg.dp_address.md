---
topic_id: vllm_ascend.arg.dp_address
canonical_term: --dp-address
topic_kind: parameter
---

# --dp-address

## Core

- topic_id: `vllm_ascend.arg.dp_address`
- canonical_term: `--dp-address`
- kind/scope: `arg` / `vllm_ascend`
- stage: `startup`
- primary_feature: `data_parallel`
- status/confidence: `needs_manual_review` / `0.86`
- source: `code` / source_tags: code
- semantics: 通过副本扩展吞吐能力，并依赖 DP 地址和 RPC 协调。
- aliases: `--dp-address`, `dp-address`, `dp_address`, `dp address`, `dpaddress`, `data_parallel`, `data parallel`, `data-parallel`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `data_parallel` 查看稳定原理。

## Deployment View

- default_behavior: IP address for data parallel master node.
- value_shape: `free_form`
- accepted_values: string value
- constraints: 错误地址/端口会导致调度与健康检查失败
- combo_effects: N/A

## Development View

- definition_ref: examples/external_online_dp/launch_online_dp.py:14
- read_ref: vllm-ascend/examples/external_online_dp/launch_online_dp.py:27, vllm-ascend/examples/external_online_dp/launch_online_dp.py:27, vllm-ascend/examples/external_online_dp/launch_online_dp.py:40
- effect_ref: vllm-ascend/examples/external_online_dp/launch_online_dp.py:27, vllm-ascend/examples/external_online_dp/launch_online_dp.py:27, vllm-ascend/examples/external_online_dp/launch_online_dp.py:40
- web_refs: 4

## Details/Edge Cases

- failure_modes: RPC 连接失败; 请求分发不均衡
- value_failure_signals: RPC 连接失败; 请求分发不均衡
- recommendation: 固定 DP 地址和端口后再迭代性能参数。
- updated_at: 2026-03-06
