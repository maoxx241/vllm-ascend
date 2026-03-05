---
topic_id: vllm_ascend.arg.tp_size
canonical_term: --tp-size
topic_kind: parameter
---

# --tp-size

## Core

- topic_id: `vllm_ascend.arg.tp_size`
- canonical_term: `--tp-size`
- kind/scope: `arg` / `vllm_ascend`
- stage: `startup`
- primary_feature: `tensor_parallel`
- status/confidence: `needs_manual_review` / `0.86`
- semantics: 按张量维度切分模型以扩展单模型可用算力。
- aliases: `--tp-size`, `tp-size`, `tp_size`, `tp size`, `tpsize`, `tensor_parallel`, `tensor parallel`, `tensor-parallel`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `tensor_parallel` 查看稳定原理。

## Deployment View

- default_behavior: Tensor parallel size
- value_shape: `numeric`
- accepted_values: int value
- constraints: 错误的通信配置会导致启动失败
- combo_effects: N/A

## Development View

- definition_ref: examples/external_online_dp/launch_online_dp.py:11, examples/offline_data_parallel.py:82, examples/offline_external_launcher.py:114
- read_ref: vllm/vllm/benchmarks/sweep/plot_pareto.py:82, vllm/vllm/benchmarks/sweep/plot_pareto.py:86, vllm/vllm/benchmarks/sweep/plot_pareto.py:87
- effect_ref: vllm/vllm/benchmarks/sweep/plot_pareto.py:86, vllm/vllm/compilation/passes/fusion/allreduce_rms_fusion.py:701, vllm/vllm/compilation/passes/fusion/collective_fusion.py:418
- web_refs: 4

## Details/Edge Cases

- failure_modes: HCCL/NCCL 初始化失败; 跨卡通信超时
- value_failure_signals: HCCL/NCCL 初始化失败; 跨卡通信超时
- recommendation: TP 变更后同步检查 max_model_len 与通信环境变量。
- updated_at: 2026-03-05
