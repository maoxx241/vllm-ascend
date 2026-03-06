---
topic_id: vllm_ascend.env.vllm_use_modelscope
canonical_term: VLLM_USE_MODELSCOPE
topic_kind: parameter
---

# VLLM_USE_MODELSCOPE

## Core

- topic_id: `vllm_ascend.env.vllm_use_modelscope`
- canonical_term: `VLLM_USE_MODELSCOPE`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `model_selection`
- status/confidence: `aligned` / `0.95`
- source: `code` / source_tags: code_reference
- semantics: 控制模型、分词器和版本选择。
- aliases: `VLLM_USE_MODELSCOPE`, `vllm_use_modelscope`, `vllm-use-modelscope`, `vllm use modelscope`, `model_selection`, `model selection`, `model-selection`

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

- definition_ref: examples/offline_data_parallel.py:67, examples/offline_external_launcher.py:76, examples/offline_weight_load.py:77
- read_ref: vllm/vllm/envs.py:18, vllm/vllm/envs.py:543, vllm/vllm/envs.py:544
- effect_ref: vllm/vllm/lora/utils.py:259, vllm/vllm/model_executor/model_loader/weight_utils.py:164, vllm/vllm/model_executor/model_loader/weight_utils.py:168
- web_refs: 5

## Details/Edge Cases

- failure_modes: 加载失败; 返回格式异常
- value_failure_signals: 加载失败; 返回格式异常
- recommendation: 固定模型版本并记录依赖。
- updated_at: 2026-03-06
