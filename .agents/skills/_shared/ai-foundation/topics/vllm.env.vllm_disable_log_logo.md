---
topic_id: vllm.env.vllm_disable_log_logo
canonical_term: VLLM_DISABLE_LOG_LOGO
topic_kind: parameter
---

# VLLM_DISABLE_LOG_LOGO

## Core

- topic_id: `vllm.env.vllm_disable_log_logo`
- canonical_term: `VLLM_DISABLE_LOG_LOGO`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `logging_debug`
- status/confidence: `aligned` / `0.91`
- semantics: 控制日志和调试可观测性。
- aliases: `VLLM_DISABLE_LOG_LOGO`, `vllm_disable_log_logo`, `vllm-disable-log-logo`, `vllm disable log logo`, `logging_debug`, `logging debug`, `logging-debug`

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

- definition_ref: vllm/envs.py:1536
- read_ref: vllm/vllm/entrypoints/utils.py:266, vllm/vllm/envs.py:232, vllm/vllm/envs.py:1536
- effect_ref: vllm/vllm/entrypoints/utils.py:266
- web_refs: 2

## Details/Edge Cases

- failure_modes: 日志过载; 关键问题难定位
- value_failure_signals: 日志过载; 关键问题难定位
- recommendation: 问题排查阶段提升日志级别，稳定后回落。
- updated_at: 2026-03-05
