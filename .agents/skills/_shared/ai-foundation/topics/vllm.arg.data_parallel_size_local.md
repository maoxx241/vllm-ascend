---
topic_id: vllm.arg.data_parallel_size_local
canonical_term: --data-parallel-size-local
topic_kind: parameter
---

# --data-parallel-size-local

## Core

- topic_id: `vllm.arg.data_parallel_size_local`
- canonical_term: `--data-parallel-size-local`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `data_parallel`
- status/confidence: `aligned` / `0.98`
- source: `code` / source_tags: code
- semantics: 通过副本扩展吞吐能力，并依赖 DP 地址和 RPC 协调。
- aliases: `--data-parallel-size-local`, `data-parallel-size-local`, `data_parallel_size_local`, `data parallel size local`, `dataparallelsizelocal`, `data_parallel`, `data parallel`, `data-parallel`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `data_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 默认 None 时按节点拓扑、后端与 LB 模式自动推导。
- value_shape: `numeric_or_auto`
- accepted_values: int >= 1, unset(auto infer)
- constraints: 必须 <= data_parallel_size。; 当 data_parallel_rank 显式给出（外部 LB）时，local size 只能是 1 或 None。; 启用 hybrid_lb 时必须可推导到有效 local size。
- combo_effects: 与 data_parallel_hybrid_lb / data_parallel_external_lb 强耦合。

## Development View

- definition_ref: vllm/engine/arg_utils.py:834
- read_ref: vllm/vllm/config/parallel.py:106, vllm/vllm/config/parallel.py:313, vllm/vllm/config/parallel.py:315
- effect_ref: vllm/vllm/config/parallel.py:313, vllm/vllm/config/parallel.py:559, vllm/vllm/engine/arg_utils.py:1534
- web_refs: 7

## Details/Edge Cases

- failure_modes: RPC 连接失败; 请求分发不均衡
- value_failure_signals: AssertionError: data_parallel_size_local must be 1 or None when data_parallel_rank is set; AssertionError: data_parallel_size_local must be set to use data_parallel_hybrid_lb
- recommendation: 固定 DP 地址和端口后再迭代性能参数。
- updated_at: 2026-03-11
