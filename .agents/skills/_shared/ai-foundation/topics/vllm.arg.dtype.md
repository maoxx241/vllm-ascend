---
topic_id: vllm.arg.dtype
canonical_term: --dtype
topic_kind: parameter
---

# --dtype

## Core

- topic_id: `vllm.arg.dtype`
- canonical_term: `--dtype`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `model_selection`
- status/confidence: `aligned` / `0.95`
- source: `code` / source_tags: code
- semantics: 控制模型、分词器和版本选择。
- aliases: `--dtype`, `dtype`, `model_selection`, `model selection`, `model-selection`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `model_selection` 查看稳定原理。

## Deployment View

- default_behavior: 默认 auto：根据模型 config dtype 与平台支持自动决策。
- value_shape: `enum`
- accepted_values: auto, half, float16, bfloat16, float, float32
- constraints: 不支持的 dtype 字符串会报错。; 部分模型类型禁用 float16（如 gemma2/gemma3/plamo2 等），会报错要求改为 bf16/float32。
- combo_effects: 与 --quantization 联动：量化未启用时，dtype 直接控制权重/激活主精度。; 与 --max-model-len/批处理参数联动影响显存峰值与可并发。

## Development View

- definition_ref: vllm/engine/arg_utils.py:655
- read_ref: vllm/vllm/_aiter_ops.py:89, vllm/vllm/_aiter_ops.py:112, vllm/vllm/_aiter_ops.py:131
- effect_ref: vllm/vllm/_aiter_ops.py:134, vllm/vllm/_aiter_ops.py:438, vllm/vllm/_aiter_ops.py:463
- web_refs: 6

## Details/Edge Cases

- failure_modes: 加载失败; 返回格式异常
- value_failure_signals: ValueError: Unknown dtype; ValueError: For Gemma 2 and 3, float16 is not supported
- recommendation: 固定模型版本并记录依赖。
- updated_at: 2026-03-06
