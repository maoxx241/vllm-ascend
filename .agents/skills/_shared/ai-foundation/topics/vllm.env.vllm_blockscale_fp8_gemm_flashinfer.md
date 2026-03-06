---
topic_id: vllm.env.vllm_blockscale_fp8_gemm_flashinfer
canonical_term: VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER
topic_kind: parameter
---

# VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER

## Core

- topic_id: `vllm.env.vllm_blockscale_fp8_gemm_flashinfer`
- canonical_term: `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `quantization`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: 选择量化实现和权重加载路径，直接影响吞吐、显存和精度。
- aliases: `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER`, `vllm_blockscale_fp8_gemm_flashinfer`, `vllm-blockscale-fp8-gemm-flashinfer`, `vllm blockscale fp8 gemm flashinfer`, `quantization`

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

- definition_ref: vllm/envs.py:1171
- read_ref: vllm/vllm/envs.py:159, vllm/vllm/envs.py:1171, vllm/vllm/envs.py:1172
- effect_ref: vllm/vllm/envs.py:159, vllm/vllm/envs.py:1171, vllm/vllm/envs.py:1172
- web_refs: 5

## Details/Edge Cases

- failure_modes: 模型加载失败; 精度异常; 推理速度低于预期
- value_failure_signals: 模型加载失败; 精度异常; 推理速度低于预期
- recommendation: 优先使用官方教程中的已验证量化工件与并行参数组合。
- updated_at: 2026-03-06
