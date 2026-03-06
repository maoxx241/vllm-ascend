---
topic_id: vllm.arg.kv_cache_dtype
canonical_term: --kv-cache-dtype
topic_kind: parameter
---

# --kv-cache-dtype

## Core

- topic_id: `vllm.arg.kv_cache_dtype`
- canonical_term: `--kv-cache-dtype`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `memory_tuning`
- status/confidence: `aligned` / `0.88`
- source: `code` / source_tags: code
- semantics: 控制 KV/权重/中间缓存占用，平衡容量与性能。
- aliases: `--kv-cache-dtype`, `kv-cache-dtype`, `kv_cache_dtype`, `kv cache dtype`, `kvcachedtype`, `memory_tuning`, `memory tuning`, `memory-tuning`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `memory_tuning` 查看稳定原理。

## Deployment View

- default_behavior: 默认 auto；在构建 CacheConfig 时会解析为具体 cache dtype。
- value_shape: `enum`
- accepted_values: auto, bfloat16, fp8, fp8_e4m3, fp8_e5m2, fp8_inc, fp8_ds_mla
- constraints: 部分 dtype 受平台后端限制（如 CUDA/ROCm/Gaudi 差异）。; 与 --calculate-kv-scales 联动决定 fp8 KV scale 来源（动态计算或读取权重）。
- combo_effects: 与 max_model_len、gpu_memory_utilization、max_num_batched_tokens 联动决定可用上下文容量。

## Development View

- definition_ref: vllm/engine/arg_utils.py:932
- read_ref: vllm/vllm/_custom_ops.py:45, vllm/vllm/_custom_ops.py:66, vllm/vllm/_custom_ops.py:92
- effect_ref: vllm/vllm/config/cache.py:106, vllm/vllm/model_executor/layers/attention/attention.py:166, vllm/vllm/model_executor/layers/attention/attention.py:358
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动或运行 OOM; 缓存不足导致吞吐下降
- value_failure_signals: 非法枚举值会在配置解析阶段报错。
- recommendation: 先保守设置，再渐进放大。
- updated_at: 2026-03-06
