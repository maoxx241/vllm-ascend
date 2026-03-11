---
topic_id: vllm.env.vllm_marlin_use_atomic_add
canonical_term: VLLM_MARLIN_USE_ATOMIC_ADD
topic_kind: parameter
---

# VLLM_MARLIN_USE_ATOMIC_ADD

## Core

- topic_id: `vllm.env.vllm_marlin_use_atomic_add`
- canonical_term: `VLLM_MARLIN_USE_ATOMIC_ADD`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `VLLM_MARLIN_USE_ATOMIC_ADD`, `vllm_marlin_use_atomic_add`, `vllm-marlin-use-atomic-add`, `vllm marlin use atomic add`, `general_runtime`, `general runtime`, `general-runtime`

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

- definition_ref: vllm/envs.py:1104
- read_ref: vllm/vllm/envs.py:142, vllm/vllm/envs.py:1104, vllm/vllm/envs.py:1105
- effect_ref: vllm/vllm/model_executor/layers/quantization/utils/marlin_utils.py:449, vllm/vllm/model_executor/layers/quantization/utils/marlin_utils.py:455, vllm/vllm/model_executor/layers/quantization/utils/marlin_utils.py:468
- web_refs: 2

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-11
