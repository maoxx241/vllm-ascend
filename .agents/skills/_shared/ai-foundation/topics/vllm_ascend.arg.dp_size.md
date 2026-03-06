---
topic_id: vllm_ascend.arg.dp_size
canonical_term: --dp-size
topic_kind: parameter
---

# --dp-size

## Core

- topic_id: `vllm_ascend.arg.dp_size`
- canonical_term: `--dp-size`
- kind/scope: `arg` / `vllm_ascend`
- stage: `startup`
- primary_feature: `data_parallel`
- status/confidence: `needs_manual_review` / `0.86`
- source: `code` / source_tags: code
- semantics: 通过副本扩展吞吐能力，并依赖 DP 地址和 RPC 协调。
- aliases: `--dp-size`, `dp-size`, `dp_size`, `dp size`, `dpsize`, `data_parallel`, `data parallel`, `data-parallel`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `data_parallel` 查看稳定原理。

## Deployment View

- default_behavior: Data parallel size
- value_shape: `numeric`
- accepted_values: int value
- constraints: 错误地址/端口会导致调度与健康检查失败
- combo_effects: N/A

## Development View

- definition_ref: examples/external_online_dp/launch_online_dp.py:10, examples/offline_data_parallel.py:81
- read_ref: vllm/vllm/benchmarks/sweep/plot_pareto.py:84, vllm/vllm/benchmarks/sweep/plot_pareto.py:90, vllm/vllm/benchmarks/sweep/plot_pareto.py:91
- effect_ref: vllm/vllm/benchmarks/sweep/plot_pareto.py:90, vllm/vllm/model_executor/layers/fused_moe/all2all_utils.py:104, vllm/vllm/model_executor/layers/fused_moe/config.py:888
- web_refs: 4

## Details/Edge Cases

- failure_modes: RPC 连接失败; 请求分发不均衡
- value_failure_signals: RPC 连接失败; 请求分发不均衡
- recommendation: 固定 DP 地址和端口后再迭代性能参数。
- updated_at: 2026-03-06
