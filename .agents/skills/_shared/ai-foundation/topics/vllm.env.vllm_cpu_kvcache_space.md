---
topic_id: vllm.env.vllm_cpu_kvcache_space
canonical_term: VLLM_CPU_KVCACHE_SPACE
topic_kind: parameter
---

# VLLM_CPU_KVCACHE_SPACE

## Core

- topic_id: `vllm.env.vllm_cpu_kvcache_space`
- canonical_term: `VLLM_CPU_KVCACHE_SPACE`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `needs_manual_review` / `0.79`
- source: `code` / source_tags: code_definition
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `VLLM_CPU_KVCACHE_SPACE`, `vllm_cpu_kvcache_space`, `vllm-cpu-kvcache-space`, `vllm cpu kvcache space`, `general_runtime`, `general runtime`, `general-runtime`

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

- definition_ref: vllm/envs.py:674
- read_ref: vllm/vllm/envs.py:50, vllm/vllm/envs.py:674, vllm/vllm/envs.py:674
- effect_ref: vllm/vllm/envs.py:675
- web_refs: 2

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-06
