---
topic_id: vllm_ascend.env.verbose
canonical_term: VERBOSE
topic_kind: parameter
---

# VERBOSE

## Core

- topic_id: `vllm_ascend.env.verbose`
- canonical_term: `VERBOSE`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `logging_debug`
- status/confidence: `aligned` / `0.88`
- semantics: 控制日志和调试可观测性。
- aliases: `VERBOSE`, `verbose`, `logging_debug`, `logging debug`, `logging-debug`

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

- definition_ref: vllm_ascend/envs.py:55
- read_ref: vllm/vllm/envs.py:84, vllm/vllm/envs.py:505, vllm/vllm/envs.py:505
- effect_ref: vllm/vllm/envs.py:84, vllm/vllm/envs.py:505, vllm/vllm/envs.py:505
- web_refs: 3

## Details/Edge Cases

- failure_modes: 日志过载; 关键问题难定位
- value_failure_signals: 日志过载; 关键问题难定位
- recommendation: 问题排查阶段提升日志级别，稳定后回落。
- updated_at: 2026-03-05
