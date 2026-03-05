---
topic_id: vllm.arg.model
canonical_term: --model
topic_kind: parameter
---

# --model

## Core

- topic_id: `vllm.arg.model`
- canonical_term: `--model`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `model_selection`
- status/confidence: `needs_manual_review` / `0.83`
- semantics: 控制模型、分词器和版本选择。
- aliases: `--model`, `model`, `model_selection`, `model selection`, `model-selection`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `model_selection` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 模型与 tokenizer/runner 不匹配
- combo_effects: N/A

## Development View

- definition_ref: vllm/engine/arg_utils.py:647
- read_ref: vllm/vllm/benchmarks/datasets.py:1569, vllm/vllm/benchmarks/datasets.py:1725, vllm/vllm/benchmarks/lib/endpoint_request_func.py:71
- effect_ref: vllm/vllm/benchmarks/lib/endpoint_request_func.py:171, vllm/vllm/benchmarks/lib/endpoint_request_func.py:300, vllm/vllm/benchmarks/lib/endpoint_request_func.py:394
- web_refs: 5

## Details/Edge Cases

- failure_modes: 加载失败; 返回格式异常
- value_failure_signals: 加载失败; 返回格式异常
- recommendation: 固定模型版本并记录依赖。
- updated_at: 2026-03-05
