---
topic_id: vllm_ascend.env.ascend_transport_print
canonical_term: ASCEND_TRANSPORT_PRINT
topic_kind: parameter
---

# ASCEND_TRANSPORT_PRINT

## Core

- topic_id: `vllm_ascend.env.ascend_transport_print`
- canonical_term: `ASCEND_TRANSPORT_PRINT`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `network_serving`
- status/confidence: `upstream_delta` / `0.55`
- source: `docs_export` / source_tags: docs_export
- semantics: 控制服务监听、路由和 API 暴露。
- aliases: `ASCEND_TRANSPORT_PRINT`, `ascend_transport_print`, `ascend-transport-print`, `ascend transport print`, `network_serving`, `network serving`, `network-serving`

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

- definition_ref: docs/source/tutorials/models/DeepSeek-V3.2.md:526, docs/source/tutorials/models/DeepSeek-V3.2.md:601, docs/source/tutorials/models/DeepSeek-V3.2.md:679
- read_ref: N/A
- effect_ref: N/A
- web_refs: 4

## Details/Edge Cases

- failure_modes: Address already in use; 健康检查 5xx
- value_failure_signals: Address already in use; 健康检查 5xx
- recommendation: 固定 host/port 并配套探活。
- updated_at: 2026-03-06
