---
topic_id: vllm.env.vllm_rocm_quick_reduce_quantization
canonical_term: VLLM_ROCM_QUICK_REDUCE_QUANTIZATION
topic_kind: parameter
---

# VLLM_ROCM_QUICK_REDUCE_QUANTIZATION

## Core

- topic_id: `vllm.env.vllm_rocm_quick_reduce_quantization`
- canonical_term: `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `quantization`
- status/confidence: `aligned` / `0.91`
- semantics: 选择量化实现和权重加载路径，直接影响吞吐、显存和精度。
- aliases: `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION`, `vllm_rocm_quick_reduce_quantization`, `vllm-rocm-quick-reduce-quantization`, `vllm rocm quick reduce quantization`, `quantization`

## Foundation

- 量化通过低比特权重/激活表示降低显存和带宽开销，常以精度换吞吐。
- 推荐结合 feature: `quantization` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `numeric`
- accepted_values: int value
- constraints: 与未验证模型配置组合时可能加载失败
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:973
- read_ref: vllm/vllm/distributed/device_communicators/quick_all_reduce.py:169, vllm/vllm/distributed/device_communicators/quick_all_reduce.py:183, vllm/vllm/envs.py:182
- effect_ref: vllm/vllm/distributed/device_communicators/quick_all_reduce.py:169, vllm/vllm/distributed/device_communicators/quick_all_reduce.py:183, vllm/vllm/envs.py:182
- web_refs: 5

## Details/Edge Cases

- failure_modes: 模型加载失败; 精度异常; 推理速度低于预期
- value_failure_signals: 模型加载失败; 精度异常; 推理速度低于预期
- recommendation: 优先使用官方教程中的已验证量化工件与并行参数组合。
- updated_at: 2026-03-05
