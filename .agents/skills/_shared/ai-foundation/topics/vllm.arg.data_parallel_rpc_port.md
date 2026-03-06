---
topic_id: vllm.arg.data_parallel_rpc_port
canonical_term: --data-parallel-rpc-port
topic_kind: parameter
---

# --data-parallel-rpc-port

## Core

- topic_id: `vllm.arg.data_parallel_rpc_port`
- canonical_term: `--data-parallel-rpc-port`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `data_parallel`
- status/confidence: `needs_manual_review` / `0.86`
- source: `code` / source_tags: code
- semantics: 通过副本扩展吞吐能力，并依赖 DP 地址和 RPC 协调。
- aliases: `--data-parallel-rpc-port`, `data-parallel-rpc-port`, `data_parallel_rpc_port`, `data parallel rpc port`, `dataparallelrpcport`, `data_parallel`, `data parallel`, `data-parallel`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `data_parallel` 查看稳定原理。

## Deployment View

- default_behavior: Port for data parallel RPC communication.
- value_shape: `numeric`
- accepted_values: int value
- constraints: 错误地址/端口会导致调度与健康检查失败
- combo_effects: N/A

## Development View

- definition_ref: vllm/engine/arg_utils.py:847
- read_ref: vllm/vllm/config/parallel.py:115, vllm/vllm/config/parallel.py:523, vllm/vllm/engine/arg_utils.py:405
- effect_ref: vllm/vllm/engine/arg_utils.py:1608, vllm/vllm/engine/arg_utils.py:1609
- web_refs: 6

## Details/Edge Cases

- failure_modes: RPC 连接失败; 请求分发不均衡
- value_failure_signals: RPC 连接失败; 请求分发不均衡
- recommendation: 固定 DP 地址和端口后再迭代性能参数。
- updated_at: 2026-03-06
