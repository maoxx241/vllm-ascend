---
topic_id: vllm.env.vllm_rocm_use_aiter_moe
canonical_term: VLLM_ROCM_USE_AITER_MOE
topic_kind: parameter
---

# VLLM_ROCM_USE_AITER_MOE

## Core

- topic_id: `vllm.env.vllm_rocm_use_aiter_moe`
- canonical_term: `VLLM_ROCM_USE_AITER_MOE`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `expert_parallel`
- status/confidence: `aligned` / `0.91`
- semantics: MoE 专家并行，提升大规模专家模型吞吐。
- aliases: `VLLM_ROCM_USE_AITER_MOE`, `vllm_rocm_use_aiter_moe`, `vllm-rocm-use-aiter-moe`, `vllm rocm use aiter moe`, `expert_parallel`, `expert parallel`, `expert-parallel`

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

- definition_ref: vllm/envs.py:901
- read_ref: vllm/vllm/_aiter_ops.py:853, vllm/vllm/_aiter_ops.py:904, vllm/vllm/_aiter_ops.py:932
- effect_ref: vllm/vllm/model_executor/layers/fused_moe/oracle/fp8.py:304, vllm/vllm/model_executor/layers/fused_moe/oracle/fp8.py:305
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动报模型不支持 EP; 专家路由异常
- value_failure_signals: 启动报模型不支持 EP; 专家路由异常
- recommendation: 仅在 MoE profile 启用，并配合 TP/DP 校验。
- updated_at: 2026-03-05
