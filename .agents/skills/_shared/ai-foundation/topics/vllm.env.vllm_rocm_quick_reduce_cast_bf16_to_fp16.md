---
topic_id: vllm.env.vllm_rocm_quick_reduce_cast_bf16_to_fp16
canonical_term: VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16
topic_kind: parameter
---

# VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16

## Core

- topic_id: `vllm.env.vllm_rocm_quick_reduce_cast_bf16_to_fp16`
- canonical_term: `VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `aligned` / `0.91`
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16`, `vllm_rocm_quick_reduce_cast_bf16_to_fp16`, `vllm-rocm-quick-reduce-cast-bf16-to-fp16`, `vllm rocm quick reduce cast bf16 to fp16`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 错误组合可能影响稳定性
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:982
- read_ref: vllm/vllm/distributed/device_communicators/quick_all_reduce.py:168, vllm/vllm/distributed/device_communicators/quick_all_reduce.py:206, vllm/vllm/envs.py:185
- effect_ref: vllm/vllm/distributed/device_communicators/quick_all_reduce.py:168, vllm/vllm/distributed/device_communicators/quick_all_reduce.py:206, vllm/vllm/envs.py:185
- web_refs: 2

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-05
