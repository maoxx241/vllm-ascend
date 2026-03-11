---
topic_id: vllm.arg.gpu_memory_utilization
canonical_term: --gpu-memory-utilization
topic_kind: parameter
---

# --gpu-memory-utilization

## Core

- topic_id: `vllm.arg.gpu_memory_utilization`
- canonical_term: `--gpu-memory-utilization`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `memory_tuning`
- status/confidence: `aligned` / `0.95`
- source: `code` / source_tags: code
- semantics: 控制 KV/权重/中间缓存占用，平衡容量与性能。
- aliases: `--gpu-memory-utilization`, `gpu-memory-utilization`, `gpu_memory_utilization`, `gpu memory utilization`, `gpumemoryutilization`, `memory_tuning`, `memory tuning`, `memory-tuning`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `memory_tuning` 查看稳定原理。

## Deployment View

- default_behavior: 默认 0.9；按实例生效。
- value_shape: `numeric_ratio`
- accepted_values: 0 < value <= 1
- constraints: 当设置 kv_cache_memory_bytes 时会忽略该参数
- combo_effects: 与 max_num_batched_tokens 联动决定 profile 峰值后可分配 KV cache

## Development View

- definition_ref: vllm/engine/arg_utils.py:924
- read_ref: vllm/vllm/config/cache.py:49, vllm/vllm/config/cache.py:165, vllm/vllm/config/cache.py:168
- effect_ref: vllm/vllm/config/cache.py:49, vllm/vllm/config/cache.py:165, vllm/vllm/config/cache.py:168
- web_refs: 6

## Details/Edge Cases

- failure_modes: 启动或运行 OOM; 缓存不足导致吞吐下降
- value_failure_signals: 运行期 OOM、服务重启或显存不足告警
- recommendation: 先保守设置，再渐进放大。
- updated_at: 2026-03-11
