---
topic_id: vllm.arg.data_parallel_start_rank
canonical_term: --data-parallel-start-rank
topic_kind: parameter
---

# --data-parallel-start-rank

## Core

- topic_id: `vllm.arg.data_parallel_start_rank`
- canonical_term: `--data-parallel-start-rank`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `data_parallel`
- status/confidence: `aligned` / `0.98`
- source: `code` / source_tags: code
- semantics: 通过副本扩展吞吐能力，并依赖 DP 地址和 RPC 协调。
- aliases: `--data-parallel-start-rank`, `data-parallel-start-rank`, `data_parallel_start_rank`, `data parallel start rank`, `dataparallelstartrank`, `data_parallel`, `data parallel`, `data-parallel`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `data_parallel` 查看稳定原理。

## Deployment View

- default_behavior: Starting data parallel rank for secondary nodes.
- value_shape: `numeric`
- accepted_values: int value
- constraints: 错误地址/端口会导致调度与健康检查失败
- combo_effects: N/A

## Development View

- definition_ref: vllm/engine/arg_utils.py:828
- read_ref: vllm/vllm/engine/arg_utils.py:402, vllm/vllm/engine/arg_utils.py:1558, vllm/vllm/engine/arg_utils.py:1577
- effect_ref: vllm/vllm/engine/arg_utils.py:1558
- web_refs: 7

## Details/Edge Cases

- failure_modes: RPC 连接失败; 请求分发不均衡
- value_failure_signals: RPC 连接失败; 请求分发不均衡
- recommendation: 固定 DP 地址和端口后再迭代性能参数。
- updated_at: 2026-03-11
