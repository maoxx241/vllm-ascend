---
topic_id: vllm_ascend.env.msmonitor_use_daemon
canonical_term: MSMONITOR_USE_DAEMON
topic_kind: parameter
---

# MSMONITOR_USE_DAEMON

## Core

- topic_id: `vllm_ascend.env.msmonitor_use_daemon`
- canonical_term: `MSMONITOR_USE_DAEMON`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `profiling_observability`
- status/confidence: `aligned` / `0.88`
- source: `code` / source_tags: code_definition, code_reference
- semantics: 控制 profiling 和 tracing 输出。
- aliases: `MSMONITOR_USE_DAEMON`, `msmonitor_use_daemon`, `msmonitor-use-daemon`, `msmonitor use daemon`, `profiling_observability`, `profiling observability`, `profiling-observability`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `profiling_observability` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `numeric`
- accepted_values: int value
- constraints: 过量采集会影响性能
- combo_effects: N/A

## Development View

- definition_ref: vllm_ascend/envs.py:91
- read_ref: vllm-ascend/vllm_ascend/envs.py:91, vllm-ascend/vllm_ascend/envs.py:91, vllm-ascend/vllm_ascend/worker/worker.py:373
- effect_ref: vllm-ascend/vllm_ascend/worker/worker.py:373, vllm-ascend/vllm_ascend/worker/worker.py:588, vllm-ascend/vllm_ascend/worker/worker.py:589
- web_refs: 3

## Details/Edge Cases

- failure_modes: 指标缺失; 追踪上报失败
- value_failure_signals: 指标缺失; 追踪上报失败
- recommendation: 按需开启细粒度 tracing，避免全量常开。
- updated_at: 2026-03-06
