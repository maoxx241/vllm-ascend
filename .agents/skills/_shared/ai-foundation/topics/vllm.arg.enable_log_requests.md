---
topic_id: vllm.arg.enable_log_requests
canonical_term: --enable-log-requests
topic_kind: parameter
---

# --enable-log-requests

## Core

- topic_id: `vllm.arg.enable_log_requests`
- canonical_term: `--enable-log-requests`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `logging_debug`
- status/confidence: `aligned` / `0.88`
- source: `code` / source_tags: code
- semantics: 控制日志和调试可观测性。
- aliases: `--enable-log-requests`, `enable-log-requests`, `enable_log_requests`, `enable log requests`, `enablelogrequests`, `logging_debug`, `logging debug`, `logging-debug`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `logging_debug` 查看稳定原理。

## Deployment View

- default_behavior: Enable logging requests.
- value_shape: `binary_or_auto`
- accepted_values: enabled, disabled, unset(auto)
- constraints: 高日志级别会增加 CPU/I/O 开销
- combo_effects: N/A

## Development View

- definition_ref: vllm/engine/arg_utils.py:2094
- read_ref: vllm/vllm/engine/arg_utils.py:2082, vllm/vllm/engine/arg_utils.py:2097, vllm/vllm/engine/arg_utils.py:2103
- effect_ref: vllm/vllm/entrypoints/openai/api_server.py:313, vllm/vllm/entrypoints/openai/cli_args.py:324, vllm/vllm/entrypoints/openai/run_batch.py:853
- web_refs: 3

## Details/Edge Cases

- failure_modes: 日志过载; 关键问题难定位
- value_failure_signals: 日志过载; 关键问题难定位
- recommendation: 问题排查阶段提升日志级别，稳定后回落。
- updated_at: 2026-03-11
