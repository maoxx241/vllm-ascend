---
topic_id: vllm.arg.show_hidden_metrics_for_version
canonical_term: --show-hidden-metrics-for-version
topic_kind: parameter
---

# --show-hidden-metrics-for-version

## Core

- topic_id: `vllm.arg.show_hidden_metrics_for_version`
- canonical_term: `--show-hidden-metrics-for-version`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `profiling_observability`
- status/confidence: `aligned` / `0.88`
- source: `code` / source_tags: code
- semantics: 控制 profiling 和 tracing 输出。
- aliases: `--show-hidden-metrics-for-version`, `show-hidden-metrics-for-version`, `show_hidden_metrics_for_version`, `show hidden metrics for version`, `showhiddenmetricsforversion`, `profiling_observability`, `profiling observability`, `profiling-observability`

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

- definition_ref: vllm/engine/arg_utils.py:1057
- read_ref: vllm/vllm/config/observability.py:21, vllm/vllm/config/observability.py:32, vllm/vllm/config/observability.py:34
- effect_ref: vllm/vllm/config/observability.py:32, vllm/vllm/config/observability.py:34
- web_refs: 3

## Details/Edge Cases

- failure_modes: 指标缺失; 追踪上报失败
- value_failure_signals: 指标缺失; 追踪上报失败
- recommendation: 按需开启细粒度 tracing，避免全量常开。
- updated_at: 2026-03-06
