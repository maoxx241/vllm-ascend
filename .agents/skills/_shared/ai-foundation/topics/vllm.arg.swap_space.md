---
topic_id: vllm.arg.swap_space
canonical_term: --swap-space
topic_kind: parameter
---

# --swap-space

## Core

- topic_id: `vllm.arg.swap_space`
- canonical_term: `--swap-space`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `memory_tuning`
- status/confidence: `upstream_delta` / `0.75`
- source: `code` / source_tags: code
- semantics: 控制 KV/权重/中间缓存占用，平衡容量与性能。
- aliases: `--swap-space`, `swap-space`, `swap_space`, `swap space`, `swapspace`, `memory_tuning`, `memory tuning`, `memory-tuning`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `memory_tuning` 查看稳定原理。

## Deployment View

- default_behavior: 默认 4 GiB / GPU。
- value_shape: `numeric`
- accepted_values: float >= 0 (GiB per GPU)
- constraints: 总 CPU swap 预算按 tensor_parallel_size 放大。; 若 swap 占用 > 70% 总内存将报错，> 40% 给 warning。
- combo_effects: 与 tensor_parallel_size 联动计算总 CPU 内存占用。

## Development View

- definition_ref: vllm/engine/arg_utils.py:930
- read_ref: vllm/vllm/config/cache.py:57, vllm/vllm/config/cache.py:197, vllm/vllm/config/cache.py:237
- effect_ref: vllm/vllm/config/cache.py:57, vllm/vllm/config/cache.py:197, vllm/vllm/config/cache.py:237
- web_refs: 4

## Details/Edge Cases

- failure_modes: 启动或运行 OOM; 缓存不足导致吞吐下降
- value_failure_signals: ValueError: Too large swap space.
- recommendation: 先保守设置，再渐进放大。
- updated_at: 2026-03-11
