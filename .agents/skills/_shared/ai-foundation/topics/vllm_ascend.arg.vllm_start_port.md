---
topic_id: vllm_ascend.arg.vllm_start_port
canonical_term: --vllm-start-port
topic_kind: parameter
---

# --vllm-start-port

## Core

- topic_id: `vllm_ascend.arg.vllm_start_port`
- canonical_term: `--vllm-start-port`
- kind/scope: `arg` / `vllm_ascend`
- stage: `startup`
- primary_feature: `network_serving`
- status/confidence: `needs_manual_review` / `0.83`
- source: `code` / source_tags: code
- semantics: 控制服务监听、路由和 API 暴露。
- aliases: `--vllm-start-port`, `vllm-start-port`, `vllm_start_port`, `vllm start port`, `vllmstartport`, `network_serving`, `network serving`, `network-serving`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `network_serving` 查看稳定原理。

## Deployment View

- default_behavior: Starting port for the engine.
- value_shape: `numeric`
- accepted_values: int value
- constraints: 端口冲突会直接启动失败
- combo_effects: N/A

## Development View

- definition_ref: examples/external_online_dp/launch_online_dp.py:16
- read_ref: vllm-ascend/examples/external_online_dp/launch_online_dp.py:29, vllm-ascend/examples/external_online_dp/launch_online_dp.py:29, vllm-ascend/examples/external_online_dp/launch_online_dp.py:57
- effect_ref: vllm-ascend/examples/external_online_dp/launch_online_dp.py:29, vllm-ascend/examples/external_online_dp/launch_online_dp.py:29, vllm-ascend/examples/external_online_dp/launch_online_dp.py:57
- web_refs: 3

## Details/Edge Cases

- failure_modes: Address already in use; 健康检查 5xx
- value_failure_signals: Address already in use; 健康检查 5xx
- recommendation: 固定 host/port 并配套探活。
- updated_at: 2026-03-11
