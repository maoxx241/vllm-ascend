---
topic_id: vllm_ascend.arg.host
canonical_term: --host
topic_kind: parameter
---

# --host

## Core

- topic_id: `vllm_ascend.arg.host`
- canonical_term: `--host`
- kind/scope: `arg` / `vllm_ascend`
- stage: `startup`
- primary_feature: `network_serving`
- status/confidence: `needs_manual_review` / `0.83`
- source: `code` / source_tags: code
- semantics: 控制服务监听、路由和 API 暴露。
- aliases: `--host`, `host`, `network_serving`, `network serving`, `network-serving`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `network_serving` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 端口冲突会直接启动失败
- combo_effects: N/A

## Development View

- definition_ref: examples/disaggregated_encoder/disagg_epd_proxy.py:698, examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py:261, examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:509
- read_ref: vllm/vllm/_aiter_ops.py:19, vllm/vllm/benchmarks/serve.py:1182, vllm/vllm/benchmarks/serve.py:1185
- effect_ref: vllm/vllm/benchmarks/serve.py:1182, vllm/vllm/benchmarks/sweep/server.py:96, vllm/vllm/config/parallel.py:207
- web_refs: 4

## Details/Edge Cases

- failure_modes: Address already in use; 健康检查 5xx
- value_failure_signals: Address already in use; 健康检查 5xx
- recommendation: 固定 host/port 并配套探活。
- updated_at: 2026-03-06
