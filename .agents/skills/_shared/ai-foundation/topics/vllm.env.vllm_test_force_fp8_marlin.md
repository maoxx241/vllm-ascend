---
topic_id: vllm.env.vllm_test_force_fp8_marlin
canonical_term: VLLM_TEST_FORCE_FP8_MARLIN
topic_kind: parameter
---

# VLLM_TEST_FORCE_FP8_MARLIN

## Core

- topic_id: `vllm.env.vllm_test_force_fp8_marlin`
- canonical_term: `VLLM_TEST_FORCE_FP8_MARLIN`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `quantization`
- status/confidence: `aligned` / `0.91`
- semantics: 选择量化实现和权重加载路径，直接影响吞吐、显存和精度。
- aliases: `VLLM_TEST_FORCE_FP8_MARLIN`, `vllm_test_force_fp8_marlin`, `vllm-test-force-fp8-marlin`, `vllm test force fp8 marlin`, `quantization`

## Foundation

- 量化通过低比特权重/激活表示降低显存和带宽开销，常以精度换吞吐。
- 推荐结合 feature: `quantization` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 与未验证模型配置组合时可能加载失败
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:825
- read_ref: vllm/vllm/envs.py:825, vllm/vllm/envs.py:826, vllm/vllm/model_executor/layers/fused_moe/oracle/fp8.py:297
- effect_ref: vllm/vllm/model_executor/layers/fused_moe/oracle/fp8.py:297, vllm/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py:231
- web_refs: 5

## Details/Edge Cases

- failure_modes: 模型加载失败; 精度异常; 推理速度低于预期
- value_failure_signals: 模型加载失败; 精度异常; 推理速度低于预期
- recommendation: 优先使用官方教程中的已验证量化工件与并行参数组合。
- updated_at: 2026-03-05
