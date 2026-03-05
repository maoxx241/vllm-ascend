---
topic_id: vllm.arg.quantization
canonical_term: --quantization
topic_kind: parameter
---

# --quantization

## Core

- topic_id: `vllm.arg.quantization`
- canonical_term: `--quantization`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `quantization`
- status/confidence: `needs_manual_review` / `0.86`
- semantics: 指定量化后端/方法（如 ascend），影响权重加载和算子路径。
- aliases: `--quantization`, `quantization`

## Foundation

- 量化通过低比特权重/激活表示降低显存和带宽开销，常以精度换吞吐。
- 推荐结合 feature: `quantization` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时优先读取模型 quantization_config；若模型未声明则按非量化路径并由 --dtype 决定精度。
- value_shape: `enum`
- accepted_values: awq, fp8, ptpc_fp8, fbgemm_fp8, fp_quant, modelopt, modelopt_fp4, modelopt_mxfp8, gguf, gptq_marlin, awq_marlin, gptq
- constraints: 若显式 --quantization 与模型 quantization_config 推断不一致，会直接报错。; 未知量化方法会报错（必须在 QUANTIZATION_METHODS 内或已注册自定义方法）。; 若方法处于 deprecated 列表且未开启 --allow-deprecated-quantization，会报错。
- combo_effects: 与 --dtype 联动：量化关闭时主要由 dtype 决定权重/激活精度。; 与模型 profile 绑定：如 qwen3-32b-w8a8 不应叠加 int4 路径（由组合规则 hard block）。

## Development View

- definition_ref: vllm/engine/arg_utils.py:670
- read_ref: vllm/vllm/_aiter_ops.py:851, vllm/vllm/_custom_ops.py:483, vllm/vllm/_custom_ops.py:494
- effect_ref: vllm/vllm/_custom_ops.py:1740, vllm/vllm/_custom_ops.py:1813, vllm/vllm/_custom_ops.py:1814
- web_refs: 8

## Details/Edge Cases

- failure_modes: 模型加载失败; 精度异常; 推理速度低于预期
- value_failure_signals: ValueError: Unknown quantization method; ValueError: Quantization method specified in model config does not match argument; ValueError: quantization method is deprecated ... set --allow-deprecated-quantization
- recommendation: 优先使用官方教程中的已验证量化工件与并行参数组合。
- updated_at: 2026-03-05
