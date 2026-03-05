---
topic_id: vllm.arg.kv_cache_metrics
canonical_term: --kv-cache-metrics
topic_kind: parameter
---

# --kv-cache-metrics

## Core

- topic_id: `vllm.arg.kv_cache_metrics`
- canonical_term: `--kv-cache-metrics`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `memory_tuning`
- status/confidence: `needs_manual_review` / `0.76`
- semantics: 控制 KV/权重/中间缓存占用，平衡容量与性能。
- aliases: `--kv-cache-metrics`, `kv-cache-metrics`, `kv_cache_metrics`, `kv cache metrics`, `kvcachemetrics`, `memory_tuning`, `memory tuning`, `memory-tuning`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `memory_tuning` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 过激参数容易触发 OOM
- combo_effects: N/A

## Development View

- definition_ref: vllm/engine/arg_utils.py:1075
- read_ref: vllm/vllm/config/observability.py:48, vllm/vllm/engine/arg_utils.py:521, vllm/vllm/engine/arg_utils.py:521
- effect_ref: vllm/vllm/v1/core/sched/scheduler.py:84
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动或运行 OOM; 缓存不足导致吞吐下降
- value_failure_signals: 启动或运行 OOM; 缓存不足导致吞吐下降
- recommendation: 先保守设置，再渐进放大。
- updated_at: 2026-03-05
