---
topic_id: vllm.env.vllm_server_dev_mode
canonical_term: VLLM_SERVER_DEV_MODE
topic_kind: parameter
---

# VLLM_SERVER_DEV_MODE

## Core

- topic_id: `vllm.env.vllm_server_dev_mode`
- canonical_term: `VLLM_SERVER_DEV_MODE`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `aligned` / `0.98`
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `VLLM_SERVER_DEV_MODE`, `vllm_server_dev_mode`, `vllm-server-dev-mode`, `vllm server dev mode`, `general_runtime`, `general runtime`, `general-runtime`

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

- definition_ref: vllm/envs.py:1012
- read_ref: vllm/vllm/benchmarks/sweep/server.py:52, vllm/vllm/benchmarks/sweep/server.py:53, vllm/vllm/entrypoints/openai/cli_args.py:206
- effect_ref: vllm/vllm/entrypoints/serve/__init__.py:13, vllm/vllm/entrypoints/serve/cache/api_router.py:70, vllm/vllm/entrypoints/serve/instrumentator/server_info.py:63
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-05
