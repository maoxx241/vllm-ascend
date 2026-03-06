---
topic_id: vllm.env.vllm_trace_function
canonical_term: VLLM_TRACE_FUNCTION
topic_kind: parameter
---

# VLLM_TRACE_FUNCTION

## Core

- topic_id: `vllm.env.vllm_trace_function`
- canonical_term: `VLLM_TRACE_FUNCTION`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `logging_debug`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: 控制日志和调试可观测性。
- aliases: `VLLM_TRACE_FUNCTION`, `vllm_trace_function`, `vllm-trace-function`, `vllm trace function`, `logging_debug`, `logging debug`, `logging-debug`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `logging_debug` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `numeric`
- accepted_values: int value
- constraints: 高日志级别会增加 CPU/I/O 开销
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:663
- read_ref: vllm/vllm/config/vllm.py:404, vllm/vllm/config/vllm.py:406, vllm/vllm/envs.py:47
- effect_ref: vllm/vllm/config/vllm.py:404, vllm/vllm/config/vllm.py:406
- web_refs: 2

## Details/Edge Cases

- failure_modes: 日志过载; 关键问题难定位
- value_failure_signals: 日志过载; 关键问题难定位
- recommendation: 问题排查阶段提升日志级别，稳定后回落。
- updated_at: 2026-03-06
