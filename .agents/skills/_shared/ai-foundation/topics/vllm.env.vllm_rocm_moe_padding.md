---
topic_id: vllm.env.vllm_rocm_moe_padding
canonical_term: VLLM_ROCM_MOE_PADDING
topic_kind: parameter
---

# VLLM_ROCM_MOE_PADDING

## Core

- topic_id: `vllm.env.vllm_rocm_moe_padding`
- canonical_term: `VLLM_ROCM_MOE_PADDING`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `expert_parallel`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: MoE 专家并行，提升大规模专家模型吞吐。
- aliases: `VLLM_ROCM_MOE_PADDING`, `vllm_rocm_moe_padding`, `vllm-rocm-moe-padding`, `vllm rocm moe padding`, `expert_parallel`, `expert parallel`, `expert-parallel`

## Foundation

- EP 面向 MoE 专家路由，Dense 模型没有专家层时不成立。
- 推荐结合 feature: `expert_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `numeric`
- accepted_values: int value
- constraints: Dense 模型不适用
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:961
- read_ref: vllm/vllm/envs.py:116, vllm/vllm/envs.py:961, vllm/vllm/envs.py:961
- effect_ref: vllm/vllm/envs.py:116, vllm/vllm/envs.py:961, vllm/vllm/envs.py:961
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动报模型不支持 EP; 专家路由异常
- value_failure_signals: 启动报模型不支持 EP; 专家路由异常
- recommendation: 仅在 MoE profile 启用，并配合 TP/DP 校验。
- updated_at: 2026-03-06
