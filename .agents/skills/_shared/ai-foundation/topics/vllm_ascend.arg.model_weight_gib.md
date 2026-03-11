---
topic_id: vllm_ascend.arg.model_weight_gib
canonical_term: --model-weight-gib
topic_kind: parameter
---

# --model-weight-gib

## Core

- topic_id: `vllm_ascend.arg.model_weight_gib`
- canonical_term: `--model-weight-gib`
- kind/scope: `arg` / `vllm_ascend`
- stage: `startup`
- primary_feature: `model_selection`
- status/confidence: `needs_manual_review` / `0.76`
- source: `code` / source_tags: code
- semantics: 控制模型、分词器和版本选择。
- aliases: `--model-weight-gib`, `model-weight-gib`, `model_weight_gib`, `model weight gib`, `modelweightgib`, `model_selection`, `model selection`, `model-selection`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `model_selection` 查看稳定原理。

## Deployment View

- default_behavior: Model weight memory usage in GiB (e.g., 1.0 for 0.5B model).
- value_shape: `numeric`
- accepted_values: float value
- constraints: 模型与 tokenizer/runner 不匹配
- combo_effects: N/A

## Development View

- definition_ref: examples/offline_external_launcher.py:129, examples/offline_weight_load.py:138
- read_ref: vllm-ascend/examples/offline_external_launcher.py:145, vllm-ascend/examples/offline_external_launcher.py:149, vllm-ascend/examples/offline_external_launcher.py:151
- effect_ref: vllm-ascend/examples/offline_external_launcher.py:145, vllm-ascend/examples/offline_external_launcher.py:149, vllm-ascend/examples/offline_external_launcher.py:151
- web_refs: 2

## Details/Edge Cases

- failure_modes: 加载失败; 返回格式异常
- value_failure_signals: 加载失败; 返回格式异常
- recommendation: 固定模型版本并记录依赖。
- updated_at: 2026-03-11
