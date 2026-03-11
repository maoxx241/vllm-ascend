---
topic_id: vllm.env.vllm_rocm_sleep_mem_chunk_size
canonical_term: VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE
topic_kind: parameter
---

# VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE

## Core

- topic_id: `vllm.env.vllm_rocm_sleep_mem_chunk_size`
- canonical_term: `VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `expert_parallel`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: MoE 专家并行，提升大规模专家模型吞吐。
- aliases: `VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE`, `vllm_rocm_sleep_mem_chunk_size`, `vllm-rocm-sleep-mem-chunk-size`, `vllm rocm sleep mem chunk size`, `expert_parallel`, `expert parallel`, `expert-parallel`

## Foundation

- EP 面向 MoE 专家路由，Dense 模型没有专家层时不成立。
- 推荐结合 feature: `expert_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `numeric`
- accepted_values: int value
- constraints: Dense 模型不适用
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:564
- read_ref: vllm/vllm/envs.py:22, vllm/vllm/envs.py:564, vllm/vllm/envs.py:565
- effect_ref: vllm/vllm/envs.py:22, vllm/vllm/envs.py:564, vllm/vllm/envs.py:565
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动报模型不支持 EP; 专家路由异常
- value_failure_signals: 启动报模型不支持 EP; 专家路由异常
- recommendation: 仅在 MoE profile 启用，并配合 TP/DP 校验。
- updated_at: 2026-03-11
