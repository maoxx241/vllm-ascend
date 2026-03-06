---
topic_id: vllm.env.vllm_moe_routing_simulation_strategy
canonical_term: VLLM_MOE_ROUTING_SIMULATION_STRATEGY
topic_kind: parameter
---

# VLLM_MOE_ROUTING_SIMULATION_STRATEGY

## Core

- topic_id: `vllm.env.vllm_moe_routing_simulation_strategy`
- canonical_term: `VLLM_MOE_ROUTING_SIMULATION_STRATEGY`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `expert_parallel`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: MoE 专家并行，提升大规模专家模型吞吐。
- aliases: `VLLM_MOE_ROUTING_SIMULATION_STRATEGY`, `vllm_moe_routing_simulation_strategy`, `vllm-moe-routing-simulation-strategy`, `vllm moe routing simulation strategy`, `expert_parallel`, `expert parallel`, `expert-parallel`

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

- definition_ref: vllm/envs.py:1277
- read_ref: vllm/vllm/envs.py:1277, vllm/vllm/envs.py:1278, vllm/vllm/model_executor/layers/fused_moe/router/router_factory.py:58
- effect_ref: vllm/vllm/model_executor/layers/fused_moe/router/router_factory.py:58
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动报模型不支持 EP; 专家路由异常
- value_failure_signals: 启动报模型不支持 EP; 专家路由异常
- recommendation: 仅在 MoE profile 启用，并配合 TP/DP 校验。
- updated_at: 2026-03-06
