---
topic_id: vllm.env.vllm_keep_alive_on_engine_death
canonical_term: VLLM_KEEP_ALIVE_ON_ENGINE_DEATH
topic_kind: parameter
---

# VLLM_KEEP_ALIVE_ON_ENGINE_DEATH

## Core

- topic_id: `vllm.env.vllm_keep_alive_on_engine_death`
- canonical_term: `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `expert_parallel`
- status/confidence: `aligned` / `0.91`
- semantics: MoE 专家并行，提升大规模专家模型吞吐。
- aliases: `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH`, `vllm_keep_alive_on_engine_death`, `vllm-keep-alive-on-engine-death`, `vllm keep alive on engine death`, `expert_parallel`, `expert parallel`, `expert-parallel`

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

- definition_ref: vllm/envs.py:812
- read_ref: vllm/vllm/entrypoints/launcher.py:149, vllm/vllm/envs.py:82, vllm/vllm/envs.py:812
- effect_ref: vllm/vllm/entrypoints/launcher.py:149
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动报模型不支持 EP; 专家路由异常
- value_failure_signals: 启动报模型不支持 EP; 专家路由异常
- recommendation: 仅在 MoE profile 启用，并配合 TP/DP 校验。
- updated_at: 2026-03-05
