---
topic_id: vllm.arg.cudagraph_metrics
canonical_term: --cudagraph-metrics
topic_kind: parameter
---

# --cudagraph-metrics

## Core

- topic_id: `vllm.arg.cudagraph_metrics`
- canonical_term: `--cudagraph-metrics`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `graph_mode`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code
- semantics: 控制 eager/graph 执行策略，通常优化吞吐与时延抖动。
- aliases: `--cudagraph-metrics`, `cudagraph-metrics`, `cudagraph_metrics`, `cudagraph metrics`, `cudagraphmetrics`, `graph_mode`, `graph mode`, `graph-mode`

## Foundation

- Ascend 图模式由 ACLGraph 与 Xlite 图配置共同作用，FULL_DECODE_ONLY 常用于先稳态加速 decode。
- 推荐结合 feature: `graph_mode` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 部分动态路径可能要求 eager 回退
- combo_effects: N/A

## Development View

- definition_ref: vllm/engine/arg_utils.py:1087
- read_ref: vllm/vllm/config/observability.py:56, vllm/vllm/engine/arg_utils.py:525, vllm/vllm/engine/arg_utils.py:525
- effect_ref: vllm/vllm/v1/metrics/loggers.py:113, vllm/vllm/v1/worker/gpu_model_runner.py:3189, vllm-ascend/vllm_ascend/worker/model_runner_v1.py:1876
- web_refs: 5

## Details/Edge Cases

- failure_modes: 图编译失败; 服务启动后首轮请求异常
- value_failure_signals: 图编译失败; 服务启动后首轮请求异常
- recommendation: 先小流量验证 FULL_DECODE_ONLY，再放量。
- updated_at: 2026-03-11
