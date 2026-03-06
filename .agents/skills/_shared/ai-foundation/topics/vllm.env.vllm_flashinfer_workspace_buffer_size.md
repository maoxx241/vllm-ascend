---
topic_id: vllm.env.vllm_flashinfer_workspace_buffer_size
canonical_term: VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE
topic_kind: parameter
---

# VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE

## Core

- topic_id: `vllm.env.vllm_flashinfer_workspace_buffer_size`
- canonical_term: `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `needs_manual_review` / `0.79`
- source: `code` / source_tags: code_definition
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE`, `vllm_flashinfer_workspace_buffer_size`, `vllm-flashinfer-workspace-buffer-size`, `vllm flashinfer workspace buffer size`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `numeric`
- accepted_values: int value
- constraints: 错误组合可能影响稳定性
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:1252
- read_ref: vllm/vllm/envs.py:167, vllm/vllm/envs.py:1252, vllm/vllm/envs.py:1253
- effect_ref: vllm/vllm/envs.py:167, vllm/vllm/envs.py:1252, vllm/vllm/envs.py:1253
- web_refs: 2

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-06
