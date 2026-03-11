---
topic_id: vllm.env.vllm_image_fetch_timeout
canonical_term: VLLM_IMAGE_FETCH_TIMEOUT
topic_kind: parameter
---

# VLLM_IMAGE_FETCH_TIMEOUT

## Core

- topic_id: `vllm.env.vllm_image_fetch_timeout`
- canonical_term: `VLLM_IMAGE_FETCH_TIMEOUT`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `multimodal`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: 控制多模态输入处理和缓存策略。
- aliases: `VLLM_IMAGE_FETCH_TIMEOUT`, `vllm_image_fetch_timeout`, `vllm-image-fetch-timeout`, `vllm image fetch timeout`, `multimodal`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `multimodal` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `numeric`
- accepted_values: int value
- constraints: 不支持多模态的模型无法启用相关参数
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:741
- read_ref: vllm/vllm/assets/base.py:37, vllm/vllm/envs.py:65, vllm/vllm/envs.py:741
- effect_ref: vllm/vllm/assets/base.py:37, vllm/vllm/envs.py:65, vllm/vllm/envs.py:741
- web_refs: 2

## Details/Edge Cases

- failure_modes: 输入解析失败; 处理时延过高
- value_failure_signals: 输入解析失败; 处理时延过高
- recommendation: 先限制每请求多模态资源，再放开。
- updated_at: 2026-03-11
