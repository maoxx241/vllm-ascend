---
topic_id: vllm.env.vllm_kv_cache_layout
canonical_term: VLLM_KV_CACHE_LAYOUT
topic_kind: parameter
---

# VLLM_KV_CACHE_LAYOUT

## Core

- topic_id: `vllm.env.vllm_kv_cache_layout`
- canonical_term: `VLLM_KV_CACHE_LAYOUT`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `memory_tuning`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: 控制 KV/权重/中间缓存占用，平衡容量与性能。
- aliases: `VLLM_KV_CACHE_LAYOUT`, `vllm_kv_cache_layout`, `vllm-kv-cache-layout`, `vllm kv cache layout`, `memory_tuning`, `memory tuning`, `memory-tuning`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `memory_tuning` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 过激参数容易触发 OOM
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:1305
- read_ref: vllm/vllm/envs.py:179, vllm/vllm/envs.py:1305, vllm/vllm/envs.py:1306
- effect_ref: vllm/vllm/envs.py:179, vllm/vllm/envs.py:1305, vllm/vllm/envs.py:1306
- web_refs: 2

## Details/Edge Cases

- failure_modes: 启动或运行 OOM; 缓存不足导致吞吐下降
- value_failure_signals: 启动或运行 OOM; 缓存不足导致吞吐下降
- recommendation: 先保守设置，再渐进放大。
- updated_at: 2026-03-06
