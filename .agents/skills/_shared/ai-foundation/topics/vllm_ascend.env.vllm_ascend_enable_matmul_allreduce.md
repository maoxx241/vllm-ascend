---
topic_id: vllm_ascend.env.vllm_ascend_enable_matmul_allreduce
canonical_term: VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE
topic_kind: parameter
---

# VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE

## Core

- topic_id: `vllm_ascend.env.vllm_ascend_enable_matmul_allreduce`
- canonical_term: `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `tensor_parallel`
- status/confidence: `aligned` / `0.98`
- source: `code` / source_tags: code_definition, code_reference
- semantics: 按张量维度切分模型以扩展单模型可用算力。
- aliases: `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE`, `vllm_ascend_enable_matmul_allreduce`, `vllm-ascend-enable-matmul-allreduce`, `vllm ascend enable matmul allreduce`, `tensor_parallel`, `tensor parallel`, `tensor-parallel`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `tensor_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `numeric`
- accepted_values: int value
- constraints: 错误的通信配置会导致启动失败
- combo_effects: N/A

## Development View

- definition_ref: vllm_ascend/batch_invariant.py:82, vllm_ascend/envs.py:71
- read_ref: vllm-ascend/vllm_ascend/batch_invariant.py:82, vllm-ascend/vllm_ascend/envs.py:71, vllm-ascend/vllm_ascend/envs.py:71
- effect_ref: vllm-ascend/vllm_ascend/utils.py:747
- web_refs: 5

## Details/Edge Cases

- failure_modes: HCCL/NCCL 初始化失败; 跨卡通信超时
- value_failure_signals: HCCL/NCCL 初始化失败; 跨卡通信超时
- recommendation: TP 变更后同步检查 max_model_len 与通信环境变量。
- updated_at: 2026-03-06
