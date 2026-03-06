---
topic_id: vllm.arg.api_server_count
canonical_term: --api-server-count
topic_kind: parameter
---

# --api-server-count

## Core

- topic_id: `vllm.arg.api_server_count`
- canonical_term: `--api-server-count`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `network_serving`
- status/confidence: `upstream_delta` / `0.75`
- source: `code` / source_tags: code
- semantics: 控制服务监听、路由和 API 暴露。
- aliases: `--api-server-count`, `api-server-count`, `api_server_count`, `api server count`, `apiservercount`, `network_serving`, `network serving`, `network-serving`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `network_serving` 查看稳定原理。

## Deployment View

- default_behavior: How many API server processes to run. Defaults to data_parallel_size if not specified.
- value_shape: `numeric`
- accepted_values: int value
- constraints: 端口冲突会直接启动失败
- combo_effects: N/A

## Development View

- definition_ref: vllm/entrypoints/openai/cli_args.py:293
- read_ref: vllm/vllm/config/multimodal.py:105, vllm/vllm/entrypoints/cli/serve.py:54, vllm/vllm/entrypoints/cli/serve.py:54
- effect_ref: vllm/vllm/entrypoints/cli/serve.py:54, vllm/vllm/entrypoints/cli/serve.py:54, vllm/vllm/entrypoints/cli/serve.py:82
- web_refs: 6

## Details/Edge Cases

- failure_modes: Address already in use; 健康检查 5xx
- value_failure_signals: Address already in use; 健康检查 5xx
- recommendation: 固定 host/port 并配套探活。
- updated_at: 2026-03-06
