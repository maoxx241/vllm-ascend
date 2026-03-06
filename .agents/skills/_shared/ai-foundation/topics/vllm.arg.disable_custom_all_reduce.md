---
topic_id: vllm.arg.disable_custom_all_reduce
canonical_term: --disable-custom-all-reduce
topic_kind: parameter
---

# --disable-custom-all-reduce

## Core

- topic_id: `vllm.arg.disable_custom_all_reduce`
- canonical_term: `--disable-custom-all-reduce`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `general_runtime`
- status/confidence: `needs_manual_review` / `0.76`
- source: `code` / source_tags: code
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `--disable-custom-all-reduce`, `disable-custom-all-reduce`, `disable_custom_all_reduce`, `disable custom all reduce`, `disablecustomallreduce`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 错误组合可能影响稳定性
- combo_effects: N/A

## Development View

- definition_ref: vllm/engine/arg_utils.py:909
- read_ref: vllm/vllm/config/parallel.py:166, vllm/vllm/config/parallel.py:530, vllm/vllm/config/parallel.py:676
- effect_ref: vllm/vllm/engine/arg_utils.py:910
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-06
