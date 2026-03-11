---
topic_id: vllm.env.vllm_rocm_fp8_mfma_page_attn
canonical_term: VLLM_ROCM_FP8_MFMA_PAGE_ATTN
topic_kind: parameter
---

# VLLM_ROCM_FP8_MFMA_PAGE_ATTN

## Core

- topic_id: `vllm.env.vllm_rocm_fp8_mfma_page_attn`
- canonical_term: `VLLM_ROCM_FP8_MFMA_PAGE_ATTN`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `quantization`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: 选择量化实现和权重加载路径，直接影响吞吐、显存和精度。
- aliases: `VLLM_ROCM_FP8_MFMA_PAGE_ATTN`, `vllm_rocm_fp8_mfma_page_attn`, `vllm-rocm-fp8-mfma-page-attn`, `vllm rocm fp8 mfma page attn`, `quantization`

## Foundation

- 量化通过低比特权重/激活表示降低显存和带宽开销，常以精度换吞吐。
- 推荐结合 feature: `quantization` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `numeric`
- accepted_values: int value
- constraints: 与未验证模型配置组合时可能加载失败
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:1414
- read_ref: vllm/vllm/_custom_ops.py:147, vllm/vllm/envs.py:202, vllm/vllm/envs.py:1414
- effect_ref: vllm/vllm/_custom_ops.py:147
- web_refs: 5

## Details/Edge Cases

- failure_modes: 模型加载失败; 精度异常; 推理速度低于预期
- value_failure_signals: 模型加载失败; 精度异常; 推理速度低于预期
- recommendation: 优先使用官方教程中的已验证量化工件与并行参数组合。
- updated_at: 2026-03-11
