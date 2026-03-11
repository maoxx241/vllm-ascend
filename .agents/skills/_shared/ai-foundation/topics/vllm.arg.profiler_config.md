---
topic_id: vllm.arg.profiler_config
canonical_term: --profiler-config
topic_kind: parameter
---

# --profiler-config

## Core

- topic_id: `vllm.arg.profiler_config`
- canonical_term: `--profiler-config`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `profiling_observability`
- status/confidence: `aligned` / `0.95`
- source: `code` / source_tags: code
- semantics: 控制 profiling 和 tracing 输出。
- aliases: `--profiler-config`, `profiler-config`, `profiler_config`, `profiler config`, `profilerconfig`, `profiling_observability`, `profiling observability`, `profiling-observability`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `profiling_observability` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 过量采集会影响性能
- combo_effects: N/A

## Development View

- definition_ref: vllm/engine/arg_utils.py:1222
- read_ref: vllm/vllm/benchmarks/latency.py:141, vllm/vllm/benchmarks/latency.py:141, vllm/vllm/benchmarks/latency.py:142
- effect_ref: vllm/vllm/benchmarks/latency.py:142, vllm/vllm/benchmarks/latency.py:147, vllm/vllm/config/vllm.py:350
- web_refs: 4

## Details/Edge Cases

- failure_modes: 指标缺失; 追踪上报失败
- value_failure_signals: 指标缺失; 追踪上报失败
- recommendation: 按需开启细粒度 tracing，避免全量常开。
- updated_at: 2026-03-11
