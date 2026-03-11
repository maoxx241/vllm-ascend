---
topic_id: vllm.arg.model_impl
canonical_term: --model-impl
topic_kind: parameter
---

# --model-impl

## Core

- topic_id: `vllm.arg.model_impl`
- canonical_term: `--model-impl`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `model_selection`
- status/confidence: `aligned` / `0.88`
- source: `code` / source_tags: code
- semantics: 控制模型、分词器和版本选择。
- aliases: `--model-impl`, `model-impl`, `model_impl`, `model impl`, `modelimpl`, `model_selection`, `model selection`, `model-selection`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `model_selection` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 模型与 tokenizer/runner 不匹配
- combo_effects: N/A

## Development View

- definition_ref: vllm/engine/arg_utils.py:721
- read_ref: vllm/vllm/config/model.py:270, vllm/vllm/config/model.py:671, vllm/vllm/engine/arg_utils.py:559
- effect_ref: vllm/vllm/model_executor/model_loader/utils.py:186, vllm/vllm/model_executor/models/registry.py:972, vllm/vllm/model_executor/models/registry.py:984
- web_refs: 3

## Details/Edge Cases

- failure_modes: 加载失败; 返回格式异常
- value_failure_signals: 加载失败; 返回格式异常
- recommendation: 固定模型版本并记录依赖。
- updated_at: 2026-03-11
