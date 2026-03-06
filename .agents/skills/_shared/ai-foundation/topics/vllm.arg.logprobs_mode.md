---
topic_id: vllm.arg.logprobs_mode
canonical_term: --logprobs-mode
topic_kind: parameter
---

# --logprobs-mode

## Core

- topic_id: `vllm.arg.logprobs_mode`
- canonical_term: `--logprobs-mode`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `logging_debug`
- status/confidence: `aligned` / `0.88`
- source: `code` / source_tags: code
- semantics: 控制日志和调试可观测性。
- aliases: `--logprobs-mode`, `logprobs-mode`, `logprobs_mode`, `logprobs mode`, `logprobsmode`, `logging_debug`, `logging debug`, `logging-debug`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `logging_debug` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 高日志级别会增加 CPU/I/O 开销
- combo_effects: N/A

## Development View

- definition_ref: vllm/engine/arg_utils.py:681
- read_ref: vllm/vllm/config/model.py:206, vllm/vllm/config/model.py:337, vllm/vllm/engine/arg_utils.py:445
- effect_ref: vllm/vllm/v1/sample/ops/topk_topp_sampler.py:106, vllm/vllm/v1/sample/ops/topk_topp_sampler.py:108, vllm/vllm/v1/sample/ops/topk_topp_sampler.py:154
- web_refs: 3

## Details/Edge Cases

- failure_modes: 日志过载; 关键问题难定位
- value_failure_signals: 日志过载; 关键问题难定位
- recommendation: 问题排查阶段提升日志级别，稳定后回落。
- updated_at: 2026-03-06
