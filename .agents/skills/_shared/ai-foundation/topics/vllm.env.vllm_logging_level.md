---
topic_id: vllm.env.vllm_logging_level
canonical_term: VLLM_LOGGING_LEVEL
topic_kind: parameter
---

# VLLM_LOGGING_LEVEL

## Core

- topic_id: `vllm.env.vllm_logging_level`
- canonical_term: `VLLM_LOGGING_LEVEL`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `logging_debug`
- status/confidence: `needs_manual_review` / `0.86`
- source: `code` / source_tags: code_definition
- semantics: 控制日志和调试可观测性。
- aliases: `VLLM_LOGGING_LEVEL`, `vllm_logging_level`, `vllm-logging-level`, `vllm logging level`, `logging_debug`, `logging debug`, `logging-debug`

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

- definition_ref: vllm/envs.py:645
- read_ref: vllm/vllm/compilation/cuda_graph.py:161, vllm/vllm/compilation/cuda_graph.py:177, vllm/vllm/config/device.py:58
- effect_ref: vllm/vllm/logger.py:247, vllm/vllm/utils/system_utils.py:74
- web_refs: 3

## Details/Edge Cases

- failure_modes: 日志过载; 关键问题难定位
- value_failure_signals: 日志过载; 关键问题难定位
- recommendation: 问题排查阶段提升日志级别，稳定后回落。
- updated_at: 2026-03-06
