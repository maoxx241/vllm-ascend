---
topic_id: vllm.env.vllm_rocm_use_aiter_fp4_asm_gemm
canonical_term: VLLM_ROCM_USE_AITER_FP4_ASM_GEMM
topic_kind: parameter
---

# VLLM_ROCM_USE_AITER_FP4_ASM_GEMM

## Core

- topic_id: `vllm.env.vllm_rocm_use_aiter_fp4_asm_gemm`
- canonical_term: `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM`, `vllm_rocm_use_aiter_fp4_asm_gemm`, `vllm-rocm-use-aiter-fp4-asm-gemm`, `vllm rocm use aiter fp4 asm gemm`, `general_runtime`, `general runtime`, `general-runtime`

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

- definition_ref: vllm/envs.py:920
- read_ref: vllm/vllm/_aiter_ops.py:858, vllm/vllm/_aiter_ops.py:913, vllm/vllm/_aiter_ops.py:939
- effect_ref: vllm/vllm/_aiter_ops.py:858, vllm/vllm/_aiter_ops.py:913, vllm/vllm/_aiter_ops.py:939
- web_refs: 2

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-06
