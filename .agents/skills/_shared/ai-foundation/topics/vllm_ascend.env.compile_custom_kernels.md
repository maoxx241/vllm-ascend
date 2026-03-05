---
topic_id: vllm_ascend.env.compile_custom_kernels
canonical_term: COMPILE_CUSTOM_KERNELS
topic_kind: parameter
---

# COMPILE_CUSTOM_KERNELS

## Core

- topic_id: `vllm_ascend.env.compile_custom_kernels`
- canonical_term: `COMPILE_CUSTOM_KERNELS`
- kind/scope: `env` / `vllm_ascend`
- stage: `build`
- primary_feature: `general_runtime`
- status/confidence: `aligned` / `0.88`
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `COMPILE_CUSTOM_KERNELS`, `compile_custom_kernels`, `compile-custom-kernels`, `compile custom kernels`, `general_runtime`, `general runtime`, `general-runtime`

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

- definition_ref: vllm_ascend/envs.py:43
- read_ref: vllm-ascend/vllm_ascend/envs.py:43, vllm-ascend/vllm_ascend/envs.py:43, vllm-ascend/vllm_ascend/worker/worker.py:88
- effect_ref: vllm-ascend/vllm_ascend/worker/worker.py:88
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-05
