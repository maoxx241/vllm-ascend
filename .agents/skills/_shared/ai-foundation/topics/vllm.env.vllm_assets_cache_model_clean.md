---
topic_id: vllm.env.vllm_assets_cache_model_clean
canonical_term: VLLM_ASSETS_CACHE_MODEL_CLEAN
topic_kind: parameter
---

# VLLM_ASSETS_CACHE_MODEL_CLEAN

## Core

- topic_id: `vllm.env.vllm_assets_cache_model_clean`
- canonical_term: `VLLM_ASSETS_CACHE_MODEL_CLEAN`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `model_selection`
- status/confidence: `aligned` / `0.91`
- semantics: 控制模型、分词器和版本选择。
- aliases: `VLLM_ASSETS_CACHE_MODEL_CLEAN`, `vllm_assets_cache_model_clean`, `vllm-assets-cache-model-clean`, `vllm assets cache model clean`, `model_selection`, `model selection`, `model-selection`

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

- definition_ref: vllm/envs.py:724
- read_ref: vllm/vllm/envs.py:64, vllm/vllm/envs.py:724, vllm/vllm/envs.py:725
- effect_ref: vllm/vllm/transformers_utils/runai_utils.py:57
- web_refs: 2

## Details/Edge Cases

- failure_modes: 加载失败; 返回格式异常
- value_failure_signals: 加载失败; 返回格式异常
- recommendation: 固定模型版本并记录依赖。
- updated_at: 2026-03-05
