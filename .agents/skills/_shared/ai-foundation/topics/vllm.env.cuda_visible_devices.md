---
topic_id: vllm.env.cuda_visible_devices
canonical_term: CUDA_VISIBLE_DEVICES
topic_kind: parameter
---

# CUDA_VISIBLE_DEVICES

## Core

- topic_id: `vllm.env.cuda_visible_devices`
- canonical_term: `CUDA_VISIBLE_DEVICES`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `CUDA_VISIBLE_DEVICES`, `cuda_visible_devices`, `cuda-visible-devices`, `cuda visible devices`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: 默认 None（由运行时/编排系统决定可见设备）。
- value_shape: `gpu_id_list`
- accepted_values: comma-separated GPU ids, e.g. 0,1,2,3, unset(None)
- constraints: 配置过窄会导致 world_size 大于可见 GPU 数并触发启动失败。; Ray 场景下该变量可能由调度器注入或在 worker 生命周期中重写。
- combo_effects: 与 tensor/pipeline/data parallel world_size 约束直接耦合。

## Development View

- definition_ref: vllm/envs.py:603
- read_ref: vllm/vllm/distributed/device_communicators/all_reduce_utils.py:99, vllm/vllm/distributed/device_communicators/all_reduce_utils.py:133, vllm/vllm/distributed/device_communicators/all_reduce_utils.py:200
- effect_ref: vllm/vllm/platforms/rocm.py:67, vllm/vllm/platforms/rocm.py:118, vllm/vllm/triton_utils/importing.py:27
- web_refs: 2

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: ValueError: World size (...) is larger than the number of available GPUs (...)
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-06
