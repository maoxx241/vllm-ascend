---
topic_id: vllm_ascend.env.master_addr
canonical_term: MASTER_ADDR
topic_kind: parameter
---

# MASTER_ADDR

## Core

- topic_id: `vllm_ascend.env.master_addr`
- canonical_term: `MASTER_ADDR`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `upstream_delta` / `0.68`
- source: `code` / source_tags: code_reference
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `MASTER_ADDR`, `master_addr`, `master-addr`, `master addr`, `general_runtime`, `general runtime`, `general-runtime`

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

- definition_ref: examples/offline_external_launcher.py:173, examples/offline_weight_load.py:174
- read_ref: vllm/vllm/v1/executor/uniproc_executor.py:171, vllm-ascend/examples/offline_external_launcher.py:173, vllm-ascend/examples/offline_weight_load.py:174
- effect_ref: vllm/vllm/v1/executor/uniproc_executor.py:171, vllm-ascend/examples/offline_external_launcher.py:173, vllm-ascend/examples/offline_weight_load.py:174
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-11
