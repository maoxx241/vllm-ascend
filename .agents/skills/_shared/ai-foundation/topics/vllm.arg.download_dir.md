---
topic_id: vllm.arg.download_dir
canonical_term: --download-dir
topic_kind: parameter
---

# --download-dir

## Core

- topic_id: `vllm.arg.download_dir`
- canonical_term: `--download-dir`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `model_selection`
- status/confidence: `needs_manual_review` / `0.76`
- source: `code` / source_tags: code
- semantics: 控制模型、分词器和版本选择。
- aliases: `--download-dir`, `download-dir`, `download_dir`, `download dir`, `downloaddir`, `model_selection`, `model selection`, `model-selection`

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

- definition_ref: vllm/engine/arg_utils.py:740
- read_ref: vllm/vllm/config/load.py:51, vllm/vllm/config/vllm.py:1453, vllm/vllm/config/vllm.py:1453
- effect_ref: vllm/vllm/config/load.py:51, vllm/vllm/config/vllm.py:1453, vllm/vllm/config/vllm.py:1453
- web_refs: 3

## Details/Edge Cases

- failure_modes: 加载失败; 返回格式异常
- value_failure_signals: 加载失败; 返回格式异常
- recommendation: 固定模型版本并记录依赖。
- updated_at: 2026-03-06
