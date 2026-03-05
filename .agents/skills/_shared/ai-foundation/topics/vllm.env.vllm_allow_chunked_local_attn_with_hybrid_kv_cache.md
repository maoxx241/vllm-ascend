---
topic_id: vllm.env.vllm_allow_chunked_local_attn_with_hybrid_kv_cache
canonical_term: VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE
topic_kind: parameter
---

# VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE

## Core

- topic_id: `vllm.env.vllm_allow_chunked_local_attn_with_hybrid_kv_cache`
- canonical_term: `VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `memory_tuning`
- status/confidence: `aligned` / `0.91`
- semantics: 控制 KV/权重/中间缓存占用，平衡容量与性能。
- aliases: `VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE`, `vllm_allow_chunked_local_attn_with_hybrid_kv_cache`, `vllm-allow-chunked-local-attn-with-hybrid-kv-cache`, `vllm allow chunked local attn with hybrid kv cache`, `memory_tuning`, `memory tuning`, `memory-tuning`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `memory_tuning` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `numeric`
- accepted_values: int value
- constraints: 过激参数容易触发 OOM
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:1386
- read_ref: vllm/vllm/config/vllm.py:1059, vllm/vllm/config/vllm.py:1064, vllm/vllm/envs.py:195
- effect_ref: vllm/vllm/config/vllm.py:1059
- web_refs: 2

## Details/Edge Cases

- failure_modes: 启动或运行 OOM; 缓存不足导致吞吐下降
- value_failure_signals: 启动或运行 OOM; 缓存不足导致吞吐下降
- recommendation: 先保守设置，再渐进放大。
- updated_at: 2026-03-05
