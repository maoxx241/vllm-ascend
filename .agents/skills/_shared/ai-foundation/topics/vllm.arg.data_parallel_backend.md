---
topic_id: vllm.arg.data_parallel_backend
canonical_term: --data-parallel-backend
topic_kind: parameter
---

# --data-parallel-backend

## Core

- topic_id: `vllm.arg.data_parallel_backend`
- canonical_term: `--data-parallel-backend`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `data_parallel`
- status/confidence: `needs_manual_review` / `0.79`
- semantics: 通过副本扩展吞吐能力，并依赖 DP 地址和 RPC 协调。
- aliases: `--data-parallel-backend`, `data-parallel-backend`, `data_parallel_backend`, `data parallel backend`, `dataparallelbackend`, `data_parallel`, `data parallel`, `data-parallel`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `data_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 默认 mp。
- value_shape: `enum`
- accepted_values: mp, ray
- constraints: nnodes > 1 仅支持 data_parallel_backend=mp。; 非法值会在 backend 选择阶段断言失败。
- combo_effects: 与 distributed_executor_backend、nnodes、data_parallel_address 推导逻辑联动。

## Development View

- definition_ref: vllm/engine/arg_utils.py:853
- read_ref: vllm/vllm/config/parallel.py:119, vllm/vllm/config/parallel.py:517, vllm/vllm/config/parallel.py:624
- effect_ref: vllm/vllm/config/parallel.py:624, vllm/vllm/engine/arg_utils.py:1574, vllm/vllm/engine/arg_utils.py:1587
- web_refs: 4

## Details/Edge Cases

- failure_modes: RPC 连接失败; 请求分发不均衡
- value_failure_signals: AssertionError: nnodes > 1 is only supported with data_parallel_backend=mp; AssertionError: data_parallel_backend can only be ray or mp
- recommendation: 固定 DP 地址和端口后再迭代性能参数。
- updated_at: 2026-03-05
