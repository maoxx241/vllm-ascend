---
topic_id: vllm.env.vllm_http_timeout_keep_alive
canonical_term: VLLM_HTTP_TIMEOUT_KEEP_ALIVE
topic_kind: parameter
---

# VLLM_HTTP_TIMEOUT_KEEP_ALIVE

## Core

- topic_id: `vllm.env.vllm_http_timeout_keep_alive`
- canonical_term: `VLLM_HTTP_TIMEOUT_KEEP_ALIVE`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `expert_parallel`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: MoE 专家并行，提升大规模专家模型吞吐。
- aliases: `VLLM_HTTP_TIMEOUT_KEEP_ALIVE`, `vllm_http_timeout_keep_alive`, `vllm-http-timeout-keep-alive`, `vllm http timeout keep alive`, `expert_parallel`, `expert parallel`, `expert-parallel`

## Foundation

- EP 面向 MoE 专家路由，Dense 模型没有专家层时不成立。
- 推荐结合 feature: `expert_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `numeric`
- accepted_values: int value
- constraints: Dense 模型不适用
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:848
- read_ref: vllm/vllm/entrypoints/api_server.py:144, vllm/vllm/entrypoints/openai/api_server.py:515, vllm/vllm/envs.py:87
- effect_ref: vllm/vllm/entrypoints/api_server.py:144, vllm/vllm/entrypoints/openai/api_server.py:515, vllm/vllm/envs.py:87
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动报模型不支持 EP; 专家路由异常
- value_failure_signals: 启动报模型不支持 EP; 专家路由异常
- recommendation: 仅在 MoE profile 启用，并配合 TP/DP 校验。
- updated_at: 2026-03-11
