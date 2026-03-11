---
topic_id: vllm.arg.disable_sliding_window
canonical_term: --disable-sliding-window
topic_kind: parameter
---

# --disable-sliding-window

## Core

- topic_id: `vllm.arg.disable_sliding_window`
- canonical_term: `--disable-sliding-window`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `general_runtime`
- status/confidence: `aligned` / `0.88`
- source: `code` / source_tags: code
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `--disable-sliding-window`, `disable-sliding-window`, `disable_sliding_window`, `disable sliding window`, `disableslidingwindow`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 错误组合可能影响稳定性
- combo_effects: N/A

## Development View

- definition_ref: vllm/engine/arg_utils.py:684
- read_ref: vllm/vllm/config/model.py:214, vllm/vllm/config/model.py:612, vllm/vllm/config/model.py:1551
- effect_ref: vllm/vllm/config/model.py:612, vllm/vllm/engine/arg_utils.py:685, vllm/vllm/engine/arg_utils.py:685
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-11
