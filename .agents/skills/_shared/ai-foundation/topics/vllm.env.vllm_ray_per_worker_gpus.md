---
topic_id: vllm.env.vllm_ray_per_worker_gpus
canonical_term: VLLM_RAY_PER_WORKER_GPUS
topic_kind: parameter
---

# VLLM_RAY_PER_WORKER_GPUS

## Core

- topic_id: `vllm.env.vllm_ray_per_worker_gpus`
- canonical_term: `VLLM_RAY_PER_WORKER_GPUS`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `aligned` / `0.91`
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `VLLM_RAY_PER_WORKER_GPUS`, `vllm_ray_per_worker_gpus`, `vllm-ray-per-worker-gpus`, `vllm ray per worker gpus`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `numeric`
- accepted_values: float value
- constraints: 错误组合可能影响稳定性
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:1029
- read_ref: vllm/vllm/envs.py:128, vllm/vllm/envs.py:1029, vllm/vllm/envs.py:1030
- effect_ref: vllm/vllm/envs.py:128, vllm/vllm/envs.py:1029, vllm/vllm/envs.py:1030
- web_refs: 2

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-05
