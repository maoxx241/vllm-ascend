---
topic_id: vllm.env.v_scale_constant
canonical_term: V_SCALE_CONSTANT
topic_kind: parameter
---

# V_SCALE_CONSTANT

## Core

- topic_id: `vllm.env.v_scale_constant`
- canonical_term: `V_SCALE_CONSTANT`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `V_SCALE_CONSTANT`, `v_scale_constant`, `v-scale-constant`, `v scale constant`, `general_runtime`, `general runtime`, `general-runtime`

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
- read_ref: vllm/vllm/envs.py:124, vllm/vllm/envs.py:1012, vllm/vllm/envs.py:1012
- effect_ref: vllm/vllm/envs.py:124, vllm/vllm/envs.py:1012, vllm/vllm/envs.py:1012
- web_refs: 2

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-11
