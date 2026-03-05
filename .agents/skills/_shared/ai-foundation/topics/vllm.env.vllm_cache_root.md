---
topic_id: vllm.env.vllm_cache_root
canonical_term: VLLM_CACHE_ROOT
topic_kind: parameter
---

# VLLM_CACHE_ROOT

## Core

- topic_id: `vllm.env.vllm_cache_root`
- canonical_term: `VLLM_CACHE_ROOT`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `aligned` / `0.91`
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `VLLM_CACHE_ROOT`, `vllm_cache_root`, `vllm-cache-root`, `vllm cache root`, `general_runtime`, `general runtime`, `general-runtime`

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

- definition_ref: vllm/envs.py:520
- read_ref: vllm/vllm/benchmarks/startup.py:43, vllm/vllm/benchmarks/startup.py:46, vllm/vllm/benchmarks/startup.py:53
- effect_ref: vllm/vllm/model_executor/models/registry.py:693
- web_refs: 2

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-05
