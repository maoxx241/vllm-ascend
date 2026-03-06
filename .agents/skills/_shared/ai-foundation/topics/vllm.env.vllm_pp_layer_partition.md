---
topic_id: vllm.env.vllm_pp_layer_partition
canonical_term: VLLM_PP_LAYER_PARTITION
topic_kind: parameter
---

# VLLM_PP_LAYER_PARTITION

## Core

- topic_id: `vllm.env.vllm_pp_layer_partition`
- canonical_term: `VLLM_PP_LAYER_PARTITION`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `needs_manual_review` / `0.79`
- source: `code` / source_tags: code_definition
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `VLLM_PP_LAYER_PARTITION`, `vllm_pp_layer_partition`, `vllm-pp-layer-partition`, `vllm pp layer partition`, `general_runtime`, `general runtime`, `general-runtime`

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

- definition_ref: vllm/envs.py:671
- read_ref: vllm/vllm/compilation/caching.py:472, vllm/vllm/distributed/utils.py:110, vllm/vllm/distributed/utils.py:132
- effect_ref: vllm/vllm/compilation/caching.py:472, vllm/vllm/distributed/utils.py:110, vllm/vllm/distributed/utils.py:132
- web_refs: 2

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-06
