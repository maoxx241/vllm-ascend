---
topic_id: vllm.env.vllm_use_fused_moe_grouped_topk
canonical_term: VLLM_USE_FUSED_MOE_GROUPED_TOPK
topic_kind: parameter
---

# VLLM_USE_FUSED_MOE_GROUPED_TOPK

## Core

- topic_id: `vllm.env.vllm_use_fused_moe_grouped_topk`
- canonical_term: `VLLM_USE_FUSED_MOE_GROUPED_TOPK`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `expert_parallel`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: MoE 专家并行，提升大规模专家模型吞吐。
- aliases: `VLLM_USE_FUSED_MOE_GROUPED_TOPK`, `vllm_use_fused_moe_grouped_topk`, `vllm-use-fused-moe-grouped-topk`, `vllm use fused moe grouped topk`, `expert_parallel`, `expert parallel`, `expert-parallel`

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

- definition_ref: vllm/envs.py:1178
- read_ref: vllm/vllm/envs.py:159, vllm/vllm/envs.py:1178, vllm/vllm/envs.py:1179
- effect_ref: vllm/vllm/envs.py:159, vllm/vllm/envs.py:1178, vllm/vllm/envs.py:1179
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动报模型不支持 EP; 专家路由异常
- value_failure_signals: 启动报模型不支持 EP; 专家路由异常
- recommendation: 仅在 MoE profile 启用，并配合 TP/DP 校验。
- updated_at: 2026-03-11
