---
topic_id: vllm.env.vllm_use_flashinfer_moe_int4
canonical_term: VLLM_USE_FLASHINFER_MOE_INT4
topic_kind: parameter
---

# VLLM_USE_FLASHINFER_MOE_INT4

## Core

- topic_id: `vllm.env.vllm_use_flashinfer_moe_int4`
- canonical_term: `VLLM_USE_FLASHINFER_MOE_INT4`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `int4_quantization`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: 启用 INT4/W4A4 量化路径，通常要求专用模型工件和内核支持。
- aliases: `VLLM_USE_FLASHINFER_MOE_INT4`, `vllm_use_flashinfer_moe_int4`, `vllm-use-flashinfer-moe-int4`, `vllm use flashinfer moe int4`, `int4_quantization`, `int4 quantization`, `int4-quantization`

## Foundation

- INT4/W4A4 需要模型工件、内核和平台三方同时支持。
- 推荐结合 feature: `int4_quantization` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `numeric`
- accepted_values: int value
- constraints: 与仅 W8A8 工件 profile 不兼容
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:1187
- read_ref: vllm/vllm/envs.py:163, vllm/vllm/envs.py:1187, vllm/vllm/envs.py:1188
- effect_ref: vllm/vllm/envs.py:163, vllm/vllm/envs.py:1187, vllm/vllm/envs.py:1188
- web_refs: 2

## Details/Edge Cases

- failure_modes: 启动时报不支持量化类型; 精度/稳定性异常
- value_failure_signals: 启动时报不支持量化类型; 精度/稳定性异常
- recommendation: 演示环境下先确认 profile 支持矩阵，再启用 int4。
- updated_at: 2026-03-06
