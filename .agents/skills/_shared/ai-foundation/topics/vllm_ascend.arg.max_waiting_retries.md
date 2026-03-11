---
topic_id: vllm_ascend.arg.max_waiting_retries
canonical_term: --max-waiting-retries
topic_kind: parameter
---

# --max-waiting-retries

## Core

- topic_id: `vllm_ascend.arg.max_waiting_retries`
- canonical_term: `--max-waiting-retries`
- kind/scope: `arg` / `vllm_ascend`
- stage: `startup`
- primary_feature: `general_runtime`
- status/confidence: `needs_manual_review` / `0.76`
- source: `code` / source_tags: code
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `--max-waiting-retries`, `max-waiting-retries`, `max_waiting_retries`, `max waiting retries`, `maxwaitingretries`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: Maximum number of retries for waiting nodes to be started
- value_shape: `numeric`
- accepted_values: int value
- constraints: 错误组合可能影响稳定性
- combo_effects: N/A

## Development View

- definition_ref: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:518
- read_ref: vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:477, vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:519
- effect_ref: vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:477
- web_refs: 2

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-11
