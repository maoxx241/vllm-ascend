---
topic_id: vllm.arg.skip_tokenizer_init
canonical_term: --skip-tokenizer-init
topic_kind: parameter
---

# --skip-tokenizer-init

## Core

- topic_id: `vllm.arg.skip_tokenizer_init`
- canonical_term: `--skip-tokenizer-init`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `model_selection`
- status/confidence: `aligned` / `0.88`
- source: `code` / source_tags: code
- semantics: 控制模型、分词器和版本选择。
- aliases: `--skip-tokenizer-init`, `skip-tokenizer-init`, `skip_tokenizer_init`, `skip tokenizer init`, `skiptokenizerinit`, `model_selection`, `model selection`, `model-selection`

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

- definition_ref: vllm/engine/arg_utils.py:688
- read_ref: vllm/vllm/config/model.py:224, vllm/vllm/config/model.py:339, vllm/vllm/config/vllm.py:1446
- effect_ref: vllm/vllm/engine/arg_utils.py:1612, vllm/vllm/entrypoints/llm.py:115, vllm/vllm/multimodal/processing/processor.py:98
- web_refs: 3

## Details/Edge Cases

- failure_modes: 加载失败; 返回格式异常
- value_failure_signals: 加载失败; 返回格式异常
- recommendation: 固定模型版本并记录依赖。
- updated_at: 2026-03-06
