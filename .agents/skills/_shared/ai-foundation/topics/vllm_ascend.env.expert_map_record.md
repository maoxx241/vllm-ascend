---
topic_id: vllm_ascend.env.expert_map_record
canonical_term: EXPERT_MAP_RECORD
topic_kind: parameter
---

# EXPERT_MAP_RECORD

## Core

- topic_id: `vllm_ascend.env.expert_map_record`
- canonical_term: `EXPERT_MAP_RECORD`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `upstream_delta` / `0.68`
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `EXPERT_MAP_RECORD`, `expert_map_record`, `expert-map-record`, `expert map record`, `general_runtime`, `general runtime`, `general-runtime`

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

- definition_ref: vllm_ascend/ascend_config.py:430, vllm_ascend/patch/platform/__init__.py:23
- read_ref: vllm-ascend/vllm_ascend/ascend_config.py:430, vllm-ascend/vllm_ascend/ascend_config.py:431, vllm-ascend/vllm_ascend/patch/platform/__init__.py:23
- effect_ref: vllm-ascend/vllm_ascend/patch/platform/__init__.py:23
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-05
