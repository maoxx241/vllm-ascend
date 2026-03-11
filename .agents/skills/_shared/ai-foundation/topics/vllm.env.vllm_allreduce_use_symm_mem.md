---
topic_id: vllm.env.vllm_allreduce_use_symm_mem
canonical_term: VLLM_ALLREDUCE_USE_SYMM_MEM
topic_kind: parameter
---

# VLLM_ALLREDUCE_USE_SYMM_MEM

## Core

- topic_id: `vllm.env.vllm_allreduce_use_symm_mem`
- canonical_term: `VLLM_ALLREDUCE_USE_SYMM_MEM`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `multimodal`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: 控制多模态输入处理和缓存策略。
- aliases: `VLLM_ALLREDUCE_USE_SYMM_MEM`, `vllm_allreduce_use_symm_mem`, `vllm-allreduce-use-symm-mem`, `vllm allreduce use symm mem`, `multimodal`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `multimodal` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `numeric`
- accepted_values: int value
- constraints: 不支持多模态的模型无法启用相关参数
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:1418
- read_ref: vllm/vllm/distributed/device_communicators/cuda_communicator.py:41, vllm/vllm/distributed/device_communicators/symm_mem.py:102, vllm/vllm/envs.py:204
- effect_ref: vllm/vllm/distributed/device_communicators/cuda_communicator.py:41, vllm/vllm/distributed/device_communicators/symm_mem.py:102, vllm/vllm/envs.py:204
- web_refs: 2

## Details/Edge Cases

- failure_modes: 输入解析失败; 处理时延过高
- value_failure_signals: 输入解析失败; 处理时延过高
- recommendation: 先限制每请求多模态资源，再放开。
- updated_at: 2026-03-11
