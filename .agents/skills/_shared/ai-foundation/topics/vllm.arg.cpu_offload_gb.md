---
topic_id: vllm.arg.cpu_offload_gb
canonical_term: --cpu-offload-gb
topic_kind: parameter
---

# --cpu-offload-gb

## Core

- topic_id: `vllm.arg.cpu_offload_gb`
- canonical_term: `--cpu-offload-gb`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `memory_tuning`
- status/confidence: `needs_manual_review` / `0.76`
- semantics: 控制 KV/权重/中间缓存占用，平衡容量与性能。
- aliases: `--cpu-offload-gb`, `cpu-offload-gb`, `cpu_offload_gb`, `cpu offload gb`, `cpuoffloadgb`, `memory_tuning`, `memory tuning`, `memory-tuning`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `memory_tuning` 查看稳定原理。

## Deployment View

- default_behavior: 默认 0（不启用 CPU offload）。
- value_shape: `numeric`
- accepted_values: float >= 0 (GiB per GPU)
- constraints: 需要较高带宽 CPU-GPU 互联，否则可能引入明显额外时延。
- combo_effects: 与 gpu_memory_utilization、max_model_len 联动影响是否能加载更大模型。

## Development View

- definition_ref: vllm/engine/arg_utils.py:946
- read_ref: vllm/vllm/config/cache.py:95, vllm/vllm/engine/arg_utils.py:436, vllm/vllm/engine/arg_utils.py:436
- effect_ref: vllm/vllm/config/cache.py:95, vllm/vllm/engine/arg_utils.py:436, vllm/vllm/engine/arg_utils.py:436
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动或运行 OOM; 缓存不足导致吞吐下降
- value_failure_signals: 配置为负值会在参数校验时报错。
- recommendation: 先保守设置，再渐进放大。
- updated_at: 2026-03-05
