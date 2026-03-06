---
topic_id: vllm.arg.limit_mm_per_prompt
canonical_term: --limit-mm-per-prompt
topic_kind: parameter
---

# --limit-mm-per-prompt

## Core

- topic_id: `vllm.arg.limit_mm_per_prompt`
- canonical_term: `--limit-mm-per-prompt`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `multimodal`
- status/confidence: `needs_manual_review` / `0.83`
- source: `code` / source_tags: code
- semantics: 控制多模态输入处理和缓存策略。
- aliases: `--limit-mm-per-prompt`, `limit-mm-per-prompt`, `limit_mm_per_prompt`, `limit mm per prompt`, `limitmmperprompt`, `multimodal`

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

- definition_ref: vllm/engine/arg_utils.py:978
- read_ref: vllm/vllm/benchmarks/datasets.py:806, vllm/vllm/benchmarks/datasets.py:950, vllm/vllm/benchmarks/datasets.py:960
- effect_ref: vllm/vllm/benchmarks/datasets.py:964, vllm/vllm/benchmarks/datasets.py:981, vllm/vllm/benchmarks/datasets.py:983
- web_refs: 5

## Details/Edge Cases

- failure_modes: 输入解析失败; 处理时延过高
- value_failure_signals: 输入解析失败; 处理时延过高
- recommendation: 先限制每请求多模态资源，再放开。
- updated_at: 2026-03-06
