---
topic_id: vllm.env.vllm_use_v2_model_runner
canonical_term: VLLM_USE_V2_MODEL_RUNNER
topic_kind: parameter
---

# VLLM_USE_V2_MODEL_RUNNER

## Core

- topic_id: `vllm.env.vllm_use_v2_model_runner`
- canonical_term: `VLLM_USE_V2_MODEL_RUNNER`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `model_selection`
- status/confidence: `needs_manual_review` / `0.79`
- source: `code` / source_tags: code_definition
- semantics: 控制模型、分词器和版本选择。
- aliases: `VLLM_USE_V2_MODEL_RUNNER`, `vllm_use_v2_model_runner`, `vllm-use-v2-model-runner`, `vllm use v2 model runner`, `model_selection`, `model selection`, `model-selection`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `model_selection` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `numeric`
- accepted_values: int value
- constraints: 模型与 tokenizer/runner 不匹配
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:1522
- read_ref: vllm/vllm/envs.py:229, vllm/vllm/envs.py:1522, vllm/vllm/envs.py:1523
- effect_ref: vllm-ascend/vllm_ascend/attention/attention_v1.py:74, vllm-ascend/vllm_ascend/attention/mla_v1.py:74, vllm-ascend/vllm_ascend/attention/sfa_v1.py:71
- web_refs: 2

## Details/Edge Cases

- failure_modes: 加载失败; 返回格式异常
- value_failure_signals: 加载失败; 返回格式异常
- recommendation: 固定模型版本并记录依赖。
- updated_at: 2026-03-06
