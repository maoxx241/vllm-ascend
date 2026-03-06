---
topic_id: vllm.arg.skip_mm_profiling
canonical_term: --skip-mm-profiling
topic_kind: parameter
---

# --skip-mm-profiling

## Core

- topic_id: `vllm.arg.skip_mm_profiling`
- canonical_term: `--skip-mm-profiling`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `multimodal`
- status/confidence: `aligned` / `0.88`
- source: `code` / source_tags: code
- semantics: 控制多模态输入处理和缓存策略。
- aliases: `--skip-mm-profiling`, `skip-mm-profiling`, `skip_mm_profiling`, `skip mm profiling`, `skipmmprofiling`, `multimodal`

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

- definition_ref: vllm/engine/arg_utils.py:1013
- read_ref: vllm/vllm/config/model.py:311, vllm/vllm/config/model.py:358, vllm/vllm/config/model.py:425
- effect_ref: vllm/vllm/v1/worker/gpu_model_runner.py:5112
- web_refs: 3

## Details/Edge Cases

- failure_modes: 输入解析失败; 处理时延过高
- value_failure_signals: 输入解析失败; 处理时延过高
- recommendation: 先限制每请求多模态资源，再放开。
- updated_at: 2026-03-06
