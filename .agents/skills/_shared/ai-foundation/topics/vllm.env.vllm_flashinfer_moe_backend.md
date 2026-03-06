---
topic_id: vllm.env.vllm_flashinfer_moe_backend
canonical_term: VLLM_FLASHINFER_MOE_BACKEND
topic_kind: parameter
---

# VLLM_FLASHINFER_MOE_BACKEND

## Core

- topic_id: `vllm.env.vllm_flashinfer_moe_backend`
- canonical_term: `VLLM_FLASHINFER_MOE_BACKEND`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `expert_parallel`
- status/confidence: `needs_manual_review` / `0.79`
- source: `code` / source_tags: code_definition
- semantics: MoE 专家并行，提升大规模专家模型吞吐。
- aliases: `VLLM_FLASHINFER_MOE_BACKEND`, `vllm_flashinfer_moe_backend`, `vllm-flashinfer-moe-backend`, `vllm flashinfer moe backend`, `expert_parallel`, `expert parallel`, `expert-parallel`

## Foundation

- EP 面向 MoE 专家路由，Dense 模型没有专家层时不成立。
- 推荐结合 feature: `expert_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: Dense 模型不适用
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:1246
- read_ref: vllm/vllm/envs.py:164, vllm/vllm/envs.py:1246, vllm/vllm/envs.py:1247
- effect_ref: vllm/vllm/model_executor/layers/fused_moe/oracle/fp8.py:219, vllm/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py:179
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动报模型不支持 EP; 专家路由异常
- value_failure_signals: 启动报模型不支持 EP; 专家路由异常
- recommendation: 仅在 MoE profile 启用，并配合 TP/DP 校验。
- updated_at: 2026-03-06
