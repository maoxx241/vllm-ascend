---
topic_id: vllm.arg.logits_processors
canonical_term: --logits-processors
topic_kind: parameter
---

# --logits-processors

## Core

- topic_id: `vllm.arg.logits_processors`
- canonical_term: `--logits-processors`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `logging_debug`
- status/confidence: `needs_manual_review` / `0.76`
- semantics: 控制日志和调试可观测性。
- aliases: `--logits-processors`, `logits-processors`, `logits_processors`, `logits processors`, `logitsprocessors`, `logging_debug`, `logging debug`, `logging-debug`

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

- definition_ref: vllm/engine/arg_utils.py:726
- read_ref: vllm/vllm/config/model.py:257, vllm/vllm/config/model.py:285, vllm/vllm/config/model.py:346
- effect_ref: vllm/vllm/entrypoints/openai/engine/protocol.py:208, vllm/vllm/sampling_params.py:584, vllm/vllm/v1/engine/input_processor.py:183
- web_refs: 3

## Details/Edge Cases

- failure_modes: 日志过载; 关键问题难定位
- value_failure_signals: 日志过载; 关键问题难定位
- recommendation: 问题排查阶段提升日志级别，稳定后回落。
- updated_at: 2026-03-05
