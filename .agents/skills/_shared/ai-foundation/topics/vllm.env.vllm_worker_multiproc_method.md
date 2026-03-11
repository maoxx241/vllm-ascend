---
topic_id: vllm.env.vllm_worker_multiproc_method
canonical_term: VLLM_WORKER_MULTIPROC_METHOD
topic_kind: parameter
---

# VLLM_WORKER_MULTIPROC_METHOD

## Core

- topic_id: `vllm.env.vllm_worker_multiproc_method`
- canonical_term: `VLLM_WORKER_MULTIPROC_METHOD`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `aligned` / `0.98`
- source: `code` / source_tags: code_definition
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `VLLM_WORKER_MULTIPROC_METHOD`, `vllm_worker_multiproc_method`, `vllm-worker-multiproc-method`, `vllm worker multiproc method`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: env 默认 fork；CLI 启动路径若未设置会主动注入 spawn。
- value_shape: `enum`
- accepted_values: fork, spawn
- constraints: 在 Ray actor、CUDA 已初始化或 WSL 场景，系统可能强制覆盖为 spawn。; Whisper 场景使用 fork 可能启动挂起并触发告警建议切 spawn。
- combo_effects: 与运行上下文（Ray/CUDA 初始化状态/WSL）联动，而非单纯静态配置。

## Development View

- definition_ref: vllm/envs.py:724
- read_ref: vllm/vllm/config/vllm.py:888, vllm/vllm/config/vllm.py:893, vllm/vllm/entrypoints/openai/api_server.py:77
- effect_ref: vllm/vllm/entrypoints/openai/api_server.py:77, vllm/vllm/entrypoints/utils.py:170, vllm/vllm/utils/system_utils.py:118
- web_refs: 4

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: warning: Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'; warning: Whisper is known to have issues with forked workers
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-11
