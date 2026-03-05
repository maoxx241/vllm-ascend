---
topic_id: vllm.env.vllm_cpu_num_of_reserved_cpu
canonical_term: VLLM_CPU_NUM_OF_RESERVED_CPU
topic_kind: parameter
---

# VLLM_CPU_NUM_OF_RESERVED_CPU

## Core

- topic_id: `vllm.env.vllm_cpu_num_of_reserved_cpu`
- canonical_term: `VLLM_CPU_NUM_OF_RESERVED_CPU`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `aligned` / `0.91`
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `VLLM_CPU_NUM_OF_RESERVED_CPU`, `vllm_cpu_num_of_reserved_cpu`, `vllm-cpu-num-of-reserved-cpu`, `vllm cpu num of reserved cpu`, `general_runtime`, `general runtime`, `general-runtime`

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

- definition_ref: vllm/envs.py:682
- read_ref: vllm/vllm/envs.py:52, vllm/vllm/envs.py:682, vllm/vllm/envs.py:683
- effect_ref: vllm/vllm/envs.py:685
- web_refs: 2

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-05
