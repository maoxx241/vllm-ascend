---
topic_id: vllm.arg.served_model_name
canonical_term: --served-model-name
topic_kind: parameter
---

# --served-model-name

## Core

- topic_id: `vllm.arg.served_model_name`
- canonical_term: `--served-model-name`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `network_serving`
- status/confidence: `needs_manual_review` / `0.83`
- source: `code` / source_tags: code
- semantics: 控制服务监听、路由和 API 暴露。
- aliases: `--served-model-name`, `served-model-name`, `served_model_name`, `served model name`, `servedmodelname`, `network_serving`, `network serving`, `network-serving`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `network_serving` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 端口冲突会直接启动失败
- combo_effects: N/A

## Development View

- definition_ref: vllm/engine/arg_utils.py:694
- read_ref: vllm/vllm/benchmarks/serve.py:1562, vllm/vllm/config/model.py:105, vllm/vllm/config/model.py:234
- effect_ref: vllm/vllm/config/model.py:1691, vllm/vllm/config/model.py:1693, vllm/vllm/config/model.py:1694
- web_refs: 4

## Details/Edge Cases

- failure_modes: Address already in use; 健康检查 5xx
- value_failure_signals: Address already in use; 健康检查 5xx
- recommendation: 固定 host/port 并配套探活。
- updated_at: 2026-03-06
