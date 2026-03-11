---
topic_id: vllm.arg.master_port
canonical_term: --master-port
topic_kind: parameter
---

# --master-port

## Core

- topic_id: `vllm.arg.master_port`
- canonical_term: `--master-port`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `network_serving`
- status/confidence: `aligned` / `0.95`
- source: `code` / source_tags: code
- semantics: 控制服务监听、路由和 API 暴露。
- aliases: `--master-port`, `master-port`, `master_port`, `master port`, `masterport`, `network_serving`, `network serving`, `network-serving`

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

- definition_ref: vllm/engine/arg_utils.py:794
- read_ref: vllm/vllm/config/parallel.py:227, vllm/vllm/config/parallel.py:526, vllm/vllm/distributed/parallel_state.py:1294
- effect_ref: vllm/vllm/config/parallel.py:227, vllm/vllm/config/parallel.py:526, vllm/vllm/distributed/parallel_state.py:1294
- web_refs: 4

## Details/Edge Cases

- failure_modes: Address already in use; 健康检查 5xx
- value_failure_signals: Address already in use; 健康检查 5xx
- recommendation: 固定 host/port 并配套探活。
- updated_at: 2026-03-11
