---
topic_id: vllm.env.vllm_logging_color
canonical_term: VLLM_LOGGING_COLOR
topic_kind: parameter
---

# VLLM_LOGGING_COLOR

## Core

- topic_id: `vllm.env.vllm_logging_color`
- canonical_term: `VLLM_LOGGING_COLOR`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `logging_debug`
- status/confidence: `aligned` / `0.91`
- semantics: 控制日志和调试可观测性。
- aliases: `VLLM_LOGGING_COLOR`, `vllm_logging_color`, `vllm-logging-color`, `vllm logging color`, `logging_debug`, `logging debug`, `logging-debug`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `logging_debug` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 高日志级别会增加 CPU/I/O 开销
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:652
- read_ref: vllm/vllm/envs.py:44, vllm/vllm/envs.py:652, vllm/vllm/envs.py:652
- effect_ref: vllm/vllm/logger.py:30, vllm/vllm/logger.py:32
- web_refs: 2

## Details/Edge Cases

- failure_modes: 日志过载; 关键问题难定位
- value_failure_signals: 日志过载; 关键问题难定位
- recommendation: 问题排查阶段提升日志级别，稳定后回落。
- updated_at: 2026-03-05
