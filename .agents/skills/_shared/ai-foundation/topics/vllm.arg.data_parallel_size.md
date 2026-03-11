---
topic_id: vllm.arg.data_parallel_size
canonical_term: --data-parallel-size
topic_kind: parameter
---

# --data-parallel-size

## Core

- topic_id: `vllm.arg.data_parallel_size`
- canonical_term: `--data-parallel-size`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `data_parallel`
- status/confidence: `aligned` / `0.98`
- source: `code` / source_tags: code
- semantics: 设置 DP 副本数，影响吞吐扩展与地址配置要求。
- aliases: `--data-parallel-size`, `data-parallel-size`, `data_parallel_size`, `data parallel size`, `dataparallelsize`, `data_parallel`, `data parallel`, `data-parallel`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `data_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 默认 1（不开启 DP）。
- value_shape: `numeric`
- accepted_values: int >= 1
- constraints: data_parallel_size_local 必须 <= data_parallel_size。; 在外部 LB 模式下，data_parallel_external_lb 仅在 data_parallel_size > 1 时允许。; 部分离线 dense 场景（env fallback）不支持/不建议 DP > 1。
- combo_effects: 与 data_parallel_backend/rank/local_size/hybrid_lb 联动决定路由与拓扑。; 与 pipeline/tensor parallel 共同决定总 world_size。

## Development View

- definition_ref: vllm/engine/arg_utils.py:818
- read_ref: vllm/vllm/benchmarks/sweep/plot_pareto.py:84, vllm/vllm/benchmarks/throughput.py:475, vllm/vllm/benchmarks/throughput.py:479
- effect_ref: vllm/vllm/benchmarks/throughput.py:483, vllm/vllm/benchmarks/throughput.py:492, vllm/vllm/benchmarks/throughput.py:662
- web_refs: 6

## Details/Edge Cases

- failure_modes: RPC 连接失败; 请求分发不均衡
- value_failure_signals: ValueError: data_parallel_size_local ... must be <= data_parallel_size; ValueError: data_parallel_external_lb can only be set when data_parallel_size > 1
- recommendation: 固定 DP 地址和端口后再迭代性能参数。
- updated_at: 2026-03-11
