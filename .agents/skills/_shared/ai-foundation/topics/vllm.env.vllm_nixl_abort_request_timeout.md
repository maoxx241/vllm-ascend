---
topic_id: vllm.env.vllm_nixl_abort_request_timeout
canonical_term: VLLM_NIXL_ABORT_REQUEST_TIMEOUT
topic_kind: parameter
---

# VLLM_NIXL_ABORT_REQUEST_TIMEOUT

## Core

- topic_id: `vllm.env.vllm_nixl_abort_request_timeout`
- canonical_term: `VLLM_NIXL_ABORT_REQUEST_TIMEOUT`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `aligned` / `0.98`
- source: `code` / source_tags: code_definition
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `VLLM_NIXL_ABORT_REQUEST_TIMEOUT`, `vllm_nixl_abort_request_timeout`, `vllm-nixl-abort-request-timeout`, `vllm nixl abort request timeout`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `numeric`
- accepted_values: int value
- constraints: 错误组合可能影响稳定性
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:1324
- read_ref: vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py:836, vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py:839, vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py:1981
- effect_ref: vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py:176
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-06
