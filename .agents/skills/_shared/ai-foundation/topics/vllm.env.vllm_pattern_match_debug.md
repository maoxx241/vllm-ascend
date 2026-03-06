---
topic_id: vllm.env.vllm_pattern_match_debug
canonical_term: VLLM_PATTERN_MATCH_DEBUG
topic_kind: parameter
---

# VLLM_PATTERN_MATCH_DEBUG

## Core

- topic_id: `vllm.env.vllm_pattern_match_debug`
- canonical_term: `VLLM_PATTERN_MATCH_DEBUG`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `logging_debug`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: 控制日志和调试可观测性。
- aliases: `VLLM_PATTERN_MATCH_DEBUG`, `vllm_pattern_match_debug`, `vllm-pattern-match-debug`, `vllm pattern match debug`, `logging_debug`, `logging debug`, `logging-debug`

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

- definition_ref: vllm/envs.py:573
- read_ref: vllm/vllm/compilation/passes/pass_manager.py:61, vllm/vllm/envs.py:217, vllm/vllm/envs.py:573
- effect_ref: vllm/vllm/compilation/passes/pass_manager.py:61
- web_refs: 2

## Details/Edge Cases

- failure_modes: 日志过载; 关键问题难定位
- value_failure_signals: 日志过载; 关键问题难定位
- recommendation: 问题排查阶段提升日志级别，稳定后回落。
- updated_at: 2026-03-06
