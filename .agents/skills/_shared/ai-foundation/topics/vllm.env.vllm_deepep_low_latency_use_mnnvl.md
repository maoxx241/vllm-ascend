---
topic_id: vllm.env.vllm_deepep_low_latency_use_mnnvl
canonical_term: VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL
topic_kind: parameter
---

# VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL

## Core

- topic_id: `vllm.env.vllm_deepep_low_latency_use_mnnvl`
- canonical_term: `VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `expert_parallel`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: MoE 专家并行，提升大规模专家模型吞吐。
- aliases: `VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL`, `vllm_deepep_low_latency_use_mnnvl`, `vllm-deepep-low-latency-use-mnnvl`, `vllm deepep low latency use mnnvl`, `expert_parallel`, `expert parallel`, `expert-parallel`

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

- definition_ref: vllm/envs.py:1467
- read_ref: vllm/vllm/distributed/device_communicators/all2all.py:480, vllm/vllm/envs.py:215, vllm/vllm/envs.py:1467
- effect_ref: vllm/vllm/distributed/device_communicators/all2all.py:480, vllm/vllm/envs.py:215, vllm/vllm/envs.py:1467
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动报模型不支持 EP; 专家路由异常
- value_failure_signals: 启动报模型不支持 EP; 专家路由异常
- recommendation: 仅在 MoE profile 启用，并配合 TP/DP 校验。
- updated_at: 2026-03-06
