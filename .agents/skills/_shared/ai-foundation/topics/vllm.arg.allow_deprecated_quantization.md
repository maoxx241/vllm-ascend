---
topic_id: vllm.arg.allow_deprecated_quantization
canonical_term: --allow-deprecated-quantization
topic_kind: parameter
---

# --allow-deprecated-quantization

## Core

- topic_id: `vllm.arg.allow_deprecated_quantization`
- canonical_term: `--allow-deprecated-quantization`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `quantization`
- status/confidence: `needs_manual_review` / `0.79`
- semantics: 选择量化实现和权重加载路径，直接影响吞吐、显存和精度。
- aliases: `--allow-deprecated-quantization`, `allow-deprecated-quantization`, `allow_deprecated_quantization`, `allow deprecated quantization`, `allowdeprecatedquantization`, `quantization`

## Foundation

- 量化通过低比特权重/激活表示降低显存和带宽开销，常以精度换吞吐。
- 推荐结合 feature: `quantization` 查看稳定原理。

## Deployment View

- default_behavior: 默认 disabled（False），遇到废弃量化方法时直接阻断。
- value_shape: `binary_toggle`
- accepted_values: enabled, disabled
- constraints: 仅在量化方法命中 deprecated 列表时生效。
- combo_effects: 与 --quantization 联动决定 deprecated 后端是告警继续还是硬失败。

## Development View

- definition_ref: vllm/engine/arg_utils.py:671
- read_ref: vllm/vllm/config/model.py:192, vllm/vllm/config/model.py:943, vllm/vllm/engine/arg_utils.py:454
- effect_ref: vllm/vllm/config/model.py:943, vllm/vllm/config/model.py:953, vllm/vllm/engine/arg_utils.py:672
- web_refs: 6

## Details/Edge Cases

- failure_modes: 模型加载失败; 精度异常; 推理速度低于预期
- value_failure_signals: ValueError: quantization method is deprecated ... set --allow-deprecated-quantization
- recommendation: 优先使用官方教程中的已验证量化工件与并行参数组合。
- updated_at: 2026-03-05
