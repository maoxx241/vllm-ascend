---
topic_id: vllm_ascend.arg.retry_delay
canonical_term: --retry-delay
topic_kind: parameter
---

# --retry-delay

## Core

- topic_id: `vllm_ascend.arg.retry_delay`
- canonical_term: `--retry-delay`
- kind/scope: `arg` / `vllm_ascend`
- stage: `startup`
- primary_feature: `general_runtime`
- status/confidence: `needs_manual_review` / `0.76`
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `--retry-delay`, `retry-delay`, `retry_delay`, `retry delay`, `retrydelay`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: Base delay (seconds) for exponential backoff retries
- value_shape: `numeric`
- accepted_values: float value
- constraints: 错误组合可能影响稳定性
- combo_effects: N/A

## Development View

- definition_ref: examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py:267, examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:515, examples/external_online_dp/dp_load_balance_proxy_server.py:188
- read_ref: vllm/vllm/transformers_utils/repo_utils.py:38, vllm/vllm/transformers_utils/repo_utils.py:50, vllm/vllm/transformers_utils/repo_utils.py:51
- effect_ref: vllm/vllm/transformers_utils/repo_utils.py:38, vllm/vllm/transformers_utils/repo_utils.py:50, vllm/vllm/transformers_utils/repo_utils.py:51
- web_refs: 2

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-05
