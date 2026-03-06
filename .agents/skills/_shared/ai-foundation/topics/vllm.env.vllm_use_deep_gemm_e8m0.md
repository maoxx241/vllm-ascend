---
topic_id: vllm.env.vllm_use_deep_gemm_e8m0
canonical_term: VLLM_USE_DEEP_GEMM_E8M0
topic_kind: parameter
---

# VLLM_USE_DEEP_GEMM_E8M0

## Core

- topic_id: `vllm.env.vllm_use_deep_gemm_e8m0`
- canonical_term: `VLLM_USE_DEEP_GEMM_E8M0`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `expert_parallel`
- status/confidence: `needs_manual_review` / `0.79`
- source: `code` / source_tags: code_definition
- semantics: MoE 专家并行，提升大规模专家模型吞吐。
- aliases: `VLLM_USE_DEEP_GEMM_E8M0`, `vllm_use_deep_gemm_e8m0`, `vllm-use-deep-gemm-e8m0`, `vllm use deep gemm e8m0`, `expert_parallel`, `expert parallel`, `expert-parallel`

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

- definition_ref: vllm/envs.py:1138
- read_ref: vllm/vllm/envs.py:151, vllm/vllm/envs.py:1138, vllm/vllm/envs.py:1139
- effect_ref: vllm/vllm/transformers_utils/config.py:689, vllm/vllm/transformers_utils/config.py:698, vllm/vllm/utils/deep_gemm.py:99
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动报模型不支持 EP; 专家路由异常
- value_failure_signals: 启动报模型不支持 EP; 专家路由异常
- recommendation: 仅在 MoE profile 启用，并配合 TP/DP 校验。
- updated_at: 2026-03-06
