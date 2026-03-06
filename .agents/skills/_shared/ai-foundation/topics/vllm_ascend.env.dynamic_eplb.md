---
topic_id: vllm_ascend.env.dynamic_eplb
canonical_term: DYNAMIC_EPLB
topic_kind: parameter
---

# DYNAMIC_EPLB

## Core

- topic_id: `vllm_ascend.env.dynamic_eplb`
- canonical_term: `DYNAMIC_EPLB`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `expert_parallel`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition, code_reference
- semantics: MoE 专家并行，提升大规模专家模型吞吐。
- aliases: `DYNAMIC_EPLB`, `dynamic_eplb`, `dynamic-eplb`, `dynamic eplb`, `expert_parallel`, `expert parallel`, `expert-parallel`

## Foundation

- EP 面向 MoE 专家路由，Dense 模型没有专家层时不成立。
- 推荐结合 feature: `expert_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `free_form`
- accepted_values: string value
- constraints: Dense 模型不适用
- combo_effects: N/A

## Development View

- definition_ref: vllm_ascend/ascend_config.py:429, vllm_ascend/envs.py:105, vllm_ascend/patch/platform/__init__.py:23
- read_ref: vllm-ascend/vllm_ascend/ascend_config.py:429, vllm-ascend/vllm_ascend/ascend_config.py:431, vllm-ascend/vllm_ascend/envs.py:105
- effect_ref: vllm-ascend/vllm_ascend/patch/platform/__init__.py:23
- web_refs: 4

## Details/Edge Cases

- failure_modes: 启动报模型不支持 EP; 专家路由异常
- value_failure_signals: 启动报模型不支持 EP; 专家路由异常
- recommendation: 仅在 MoE profile 启用，并配合 TP/DP 校验。
- updated_at: 2026-03-06
