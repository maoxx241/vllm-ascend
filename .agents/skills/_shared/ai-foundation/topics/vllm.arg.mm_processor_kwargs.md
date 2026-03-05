---
topic_id: vllm.arg.mm_processor_kwargs
canonical_term: --mm-processor-kwargs
topic_kind: parameter
---

# --mm-processor-kwargs

## Core

- topic_id: `vllm.arg.mm_processor_kwargs`
- canonical_term: `--mm-processor-kwargs`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `multimodal`
- status/confidence: `needs_manual_review` / `0.76`
- semantics: 控制多模态输入处理和缓存策略。
- aliases: `--mm-processor-kwargs`, `mm-processor-kwargs`, `mm_processor_kwargs`, `mm processor kwargs`, `mmprocessorkwargs`, `multimodal`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `multimodal` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 不支持多模态的模型无法启用相关参数
- combo_effects: N/A

## Development View

- definition_ref: vllm/engine/arg_utils.py:987
- read_ref: vllm/vllm/beam_search.py:31, vllm/vllm/config/model.py:303, vllm/vllm/config/model.py:352
- effect_ref: vllm/vllm/entrypoints/llm.py:656, vllm/vllm/entrypoints/llm.py:676, vllm/vllm/entrypoints/pooling/score/serving.py:404
- web_refs: 3

## Details/Edge Cases

- failure_modes: 输入解析失败; 处理时延过高
- value_failure_signals: 输入解析失败; 处理时延过高
- recommendation: 先限制每请求多模态资源，再放开。
- updated_at: 2026-03-05
