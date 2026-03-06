---
topic_id: vllm.env.vllm_use_nccl_symm_mem
canonical_term: VLLM_USE_NCCL_SYMM_MEM
topic_kind: parameter
---

# VLLM_USE_NCCL_SYMM_MEM

## Core

- topic_id: `vllm.env.vllm_use_nccl_symm_mem`
- canonical_term: `VLLM_USE_NCCL_SYMM_MEM`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `multimodal`
- status/confidence: `needs_manual_review` / `0.79`
- source: `code` / source_tags: code_definition
- semantics: 控制多模态输入处理和缓存策略。
- aliases: `VLLM_USE_NCCL_SYMM_MEM`, `vllm_use_nccl_symm_mem`, `vllm-use-nccl-symm-mem`, `vllm use nccl symm mem`, `multimodal`

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

- definition_ref: vllm/envs.py:1485
- read_ref: vllm/vllm/distributed/device_communicators/pynccl_allocator.py:50, vllm/vllm/distributed/device_communicators/pynccl_wrapper.py:350, vllm/vllm/distributed/device_communicators/pynccl_wrapper.py:353
- effect_ref: vllm/vllm/distributed/device_communicators/pynccl_allocator.py:50, vllm/vllm/distributed/device_communicators/pynccl_wrapper.py:350, vllm/vllm/distributed/device_communicators/pynccl_wrapper.py:353
- web_refs: 2

## Details/Edge Cases

- failure_modes: 输入解析失败; 处理时延过高
- value_failure_signals: 输入解析失败; 处理时延过高
- recommendation: 先限制每请求多模态资源，再放开。
- updated_at: 2026-03-06
