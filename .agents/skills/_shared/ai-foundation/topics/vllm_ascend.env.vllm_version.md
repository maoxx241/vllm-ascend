---
topic_id: vllm_ascend.env.vllm_version
canonical_term: VLLM_VERSION
topic_kind: parameter
---

# VLLM_VERSION

## Core

- topic_id: `vllm_ascend.env.vllm_version`
- canonical_term: `VLLM_VERSION`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `aligned` / `0.95`
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `VLLM_VERSION`, `vllm_version`, `vllm-version`, `vllm version`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 错误组合可能影响稳定性
- combo_effects: N/A

## Development View

- definition_ref: vllm_ascend/envs.py:68
- read_ref: vllm/vllm/connections.py:11, vllm/vllm/connections.py:48, vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py:58
- effect_ref: vllm/vllm/connections.py:48, vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py:770, vllm-ascend/vllm_ascend/utils.py:384
- web_refs: 6

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-05
