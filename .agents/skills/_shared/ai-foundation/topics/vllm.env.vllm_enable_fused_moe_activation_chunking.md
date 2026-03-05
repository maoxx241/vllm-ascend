---
topic_id: vllm.env.vllm_enable_fused_moe_activation_chunking
canonical_term: VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING
topic_kind: parameter
---

# VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING

## Core

- topic_id: `vllm.env.vllm_enable_fused_moe_activation_chunking`
- canonical_term: `VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `expert_parallel`
- status/confidence: `aligned` / `0.91`
- semantics: MoE 专家并行，提升大规模专家模型吞吐。
- aliases: `VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING`, `vllm_enable_fused_moe_activation_chunking`, `vllm-enable-fused-moe-activation-chunking`, `vllm enable fused moe activation chunking`, `expert_parallel`, `expert parallel`, `expert-parallel`

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

- definition_ref: vllm/envs.py:807
- read_ref: vllm/vllm/envs.py:57, vllm/vllm/envs.py:807, vllm/vllm/envs.py:808
- effect_ref: vllm/vllm/envs.py:57, vllm/vllm/envs.py:807, vllm/vllm/envs.py:808
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动报模型不支持 EP; 专家路由异常
- value_failure_signals: 启动报模型不支持 EP; 专家路由异常
- recommendation: 仅在 MoE profile 启用，并配合 TP/DP 校验。
- updated_at: 2026-03-05
