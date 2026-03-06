---
topic_id: vllm.env.vllm_marlin_input_dtype
canonical_term: VLLM_MARLIN_INPUT_DTYPE
topic_kind: parameter
---

# VLLM_MARLIN_INPUT_DTYPE

## Core

- topic_id: `vllm.env.vllm_marlin_input_dtype`
- canonical_term: `VLLM_MARLIN_INPUT_DTYPE`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `model_selection`
- status/confidence: `needs_manual_review` / `0.79`
- source: `code` / source_tags: code_definition
- semantics: 控制模型、分词器和版本选择。
- aliases: `VLLM_MARLIN_INPUT_DTYPE`, `vllm_marlin_input_dtype`, `vllm-marlin-input-dtype`, `vllm marlin input dtype`, `model_selection`, `model selection`, `model-selection`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `model_selection` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `numeric`
- accepted_values: int value
- constraints: 模型与 tokenizer/runner 不匹配
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:1101
- read_ref: vllm/vllm/envs.py:142, vllm/vllm/envs.py:1101, vllm/vllm/envs.py:1102
- effect_ref: vllm/vllm/model_executor/layers/quantization/utils/marlin_utils.py:493, vllm/vllm/model_executor/layers/quantization/utils/marlin_utils.py:495, vllm/vllm/model_executor/layers/quantization/utils/marlin_utils.py:497
- web_refs: 2

## Details/Edge Cases

- failure_modes: 加载失败; 返回格式异常
- value_failure_signals: 加载失败; 返回格式异常
- recommendation: 固定模型版本并记录依赖。
- updated_at: 2026-03-06
