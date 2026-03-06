---
topic_id: vllm.env.vllm_execute_model_timeout_seconds
canonical_term: VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS
topic_kind: parameter
---

# VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS

## Core

- topic_id: `vllm.env.vllm_execute_model_timeout_seconds`
- canonical_term: `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `model_selection`
- status/confidence: `aligned` / `0.98`
- source: `code` / source_tags: code_definition
- semantics: 控制模型、分词器和版本选择。
- aliases: `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`, `vllm_execute_model_timeout_seconds`, `vllm-execute-model-timeout-seconds`, `vllm execute model timeout seconds`, `model_selection`, `model selection`, `model-selection`

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

- definition_ref: vllm/envs.py:1295
- read_ref: vllm/vllm/envs.py:178, vllm/vllm/envs.py:1295, vllm/vllm/envs.py:1296
- effect_ref: vllm/vllm/envs.py:178, vllm/vllm/envs.py:1295, vllm/vllm/envs.py:1296
- web_refs: 3

## Details/Edge Cases

- failure_modes: 加载失败; 返回格式异常
- value_failure_signals: 加载失败; 返回格式异常
- recommendation: 固定模型版本并记录依赖。
- updated_at: 2026-03-06
