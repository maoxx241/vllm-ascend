---
topic_id: vllm.arg.all2all_backend
canonical_term: --all2all-backend
topic_kind: parameter
---

# --all2all-backend

## Core

- topic_id: `vllm.arg.all2all_backend`
- canonical_term: `--all2all-backend`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `tensor_parallel`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code
- semantics: 按张量维度切分模型以扩展单模型可用算力。
- aliases: `--all2all-backend`, `all2all-backend`, `all2all_backend`, `all2all backend`, `all2allbackend`, `tensor_parallel`, `tensor parallel`, `tensor-parallel`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `tensor_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 默认 allgather_reducescatter。
- value_shape: `enum`
- accepted_values: naive, pplx, deepep_high_throughput, deepep_low_latency, mori, allgather_reducescatter, flashinfer_all2allv
- constraints: 主要在 enable_expert_parallel 场景生效。
- combo_effects: 与 enable_expert_parallel + TP>1 + DP>1 时，部分后端会触发 sequence parallel MoE 路径。

## Development View

- definition_ref: vllm/engine/arg_utils.py:874
- read_ref: vllm/vllm/config/compilation.py:938, vllm/vllm/config/compilation.py:1005, vllm/vllm/config/parallel.py:150
- effect_ref: vllm/vllm/distributed/device_communicators/cpu_communicator.py:42, vllm/vllm/distributed/device_communicators/cpu_communicator.py:49, vllm/vllm/distributed/device_communicators/cuda_communicator.py:93
- web_refs: 4

## Details/Edge Cases

- failure_modes: HCCL/NCCL 初始化失败; 跨卡通信超时
- value_failure_signals: 非法枚举值会在参数解析阶段报错。
- recommendation: TP 变更后同步检查 max_model_len 与通信环境变量。
- updated_at: 2026-03-11
