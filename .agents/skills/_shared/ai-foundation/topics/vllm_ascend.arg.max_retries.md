---
topic_id: vllm_ascend.arg.max_retries
canonical_term: --max-retries
topic_kind: parameter
---

# --max-retries

## Core

- topic_id: `vllm_ascend.arg.max_retries`
- canonical_term: `--max-retries`
- kind/scope: `arg` / `vllm_ascend`
- stage: `startup`
- primary_feature: `general_runtime`
- status/confidence: `needs_manual_review` / `0.76`
- source: `code` / source_tags: code
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `--max-retries`, `max-retries`, `max_retries`, `max retries`, `maxretries`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: Maximum number of retries for HTTP requests
- value_shape: `numeric`
- accepted_values: int value
- constraints: 错误组合可能影响稳定性
- combo_effects: N/A

## Development View

- definition_ref: examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py:266, examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:514, examples/external_online_dp/dp_load_balance_proxy_server.py:187
- read_ref: vllm/vllm/config/parallel.py:413, vllm/vllm/config/parallel.py:415, vllm/vllm/distributed/eplb/eplb_state.py:1007
- effect_ref: vllm/vllm/distributed/eplb/eplb_state.py:1011, vllm/vllm/entrypoints/openai/run_batch.py:402, vllm/vllm/transformers_utils/repo_utils.py:44
- web_refs: 2

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-11
