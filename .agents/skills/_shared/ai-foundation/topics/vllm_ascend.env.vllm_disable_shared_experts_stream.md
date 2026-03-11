---
topic_id: vllm_ascend.env.vllm_disable_shared_experts_stream
canonical_term: VLLM_DISABLE_SHARED_EXPERTS_STREAM
topic_kind: parameter
---

# VLLM_DISABLE_SHARED_EXPERTS_STREAM

## Core

- topic_id: `vllm_ascend.env.vllm_disable_shared_experts_stream`
- canonical_term: `VLLM_DISABLE_SHARED_EXPERTS_STREAM`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `aligned` / `0.88`
- source: `code` / source_tags: code_reference
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `VLLM_DISABLE_SHARED_EXPERTS_STREAM`, `vllm_disable_shared_experts_stream`, `vllm-disable-shared-experts-stream`, `vllm disable shared experts stream`, `general_runtime`, `general runtime`, `general-runtime`

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

- definition_ref: vllm_ascend/platform.py:31
- read_ref: vllm/vllm/envs.py:227, vllm/vllm/envs.py:1514, vllm/vllm/envs.py:1515
- effect_ref: vllm/vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py:183
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-11
