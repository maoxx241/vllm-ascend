---
topic_id: vllm.arg.block_size
canonical_term: --block-size
topic_kind: parameter
---

# --block-size

## Core

- topic_id: `vllm.arg.block_size`
- canonical_term: `--block-size`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `memory_tuning`
- status/confidence: `needs_manual_review` / `0.83`
- semantics: 控制 KV/权重/中间缓存占用，平衡容量与性能。
- aliases: `--block-size`, `block-size`, `block_size`, `block size`, `blocksize`, `memory_tuning`, `memory tuning`, `memory-tuning`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `memory_tuning` 查看稳定原理。

## Deployment View

- default_behavior: 平台侧会在未设置时选择合适默认值。
- value_shape: `enum_numeric`
- accepted_values: 1, 8, 16, 32, 64, 128, 256
- constraints: CUDA 通常仅支持 <=32; Xlite graph 要求 block_size = 128; CP interleave 需满足 block_size 可整除 cp_kv_cache_interleave_size
- combo_effects: 与 attention backend 支持的 kernel block size 联动; 与 context parallel / graph mode 存在硬约束

## Development View

- definition_ref: vllm/engine/arg_utils.py:924
- read_ref: vllm/vllm/_aiter_ops.py:1257, vllm/vllm/_aiter_ops.py:1270, vllm/vllm/_custom_ops.py:42
- effect_ref: vllm/vllm/config/vllm.py:1111, vllm/vllm/distributed/kv_transfer/kv_connector/utils.py:429, vllm/vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:450
- web_refs: 5

## Details/Edge Cases

- failure_modes: 启动或运行 OOM; 缓存不足导致吞吐下降
- value_failure_signals: RuntimeError: Xlite graph mode is only compatible with block_size of 128; AssertionError: block_size should be divisible by cp_kv_cache_interleave_size
- recommendation: 先保守设置，再渐进放大。
- updated_at: 2026-03-05
