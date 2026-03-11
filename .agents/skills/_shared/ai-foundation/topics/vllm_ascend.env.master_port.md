---
topic_id: vllm_ascend.env.master_port
canonical_term: MASTER_PORT
topic_kind: parameter
---

# MASTER_PORT

## Core

- topic_id: `vllm_ascend.env.master_port`
- canonical_term: `MASTER_PORT`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `network_serving`
- status/confidence: `upstream_delta` / `0.68`
- source: `code` / source_tags: code_reference
- semantics: 控制服务监听、路由和 API 暴露。
- aliases: `MASTER_PORT`, `master_port`, `master-port`, `master port`, `network_serving`, `network serving`, `network-serving`

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

- definition_ref: examples/offline_external_launcher.py:174, examples/offline_weight_load.py:175
- read_ref: vllm/vllm/v1/executor/uniproc_executor.py:172, vllm-ascend/examples/offline_external_launcher.py:174, vllm-ascend/examples/offline_weight_load.py:175
- effect_ref: vllm/vllm/v1/executor/uniproc_executor.py:172, vllm-ascend/examples/offline_external_launcher.py:174, vllm-ascend/examples/offline_weight_load.py:175
- web_refs: 3

## Details/Edge Cases

- failure_modes: Address already in use; 健康检查 5xx
- value_failure_signals: Address already in use; 健康检查 5xx
- recommendation: 固定 host/port 并配套探活。
- updated_at: 2026-03-11
