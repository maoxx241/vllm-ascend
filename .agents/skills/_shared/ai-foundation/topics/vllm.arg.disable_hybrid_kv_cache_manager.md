---
topic_id: vllm.arg.disable_hybrid_kv_cache_manager
canonical_term: --disable-hybrid-kv-cache-manager
topic_kind: parameter
---

# --disable-hybrid-kv-cache-manager

## Core

- topic_id: `vllm.arg.disable_hybrid_kv_cache_manager`
- canonical_term: `--disable-hybrid-kv-cache-manager`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `memory_tuning`
- status/confidence: `needs_manual_review` / `0.76`
- semantics: 控制 KV/权重/中间缓存占用，平衡容量与性能。
- aliases: `--disable-hybrid-kv-cache-manager`, `disable-hybrid-kv-cache-manager`, `disable_hybrid_kv_cache_manager`, `disable hybrid kv cache manager`, `disablehybridkvcachemanager`, `memory_tuning`, `memory tuning`, `memory-tuning`

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

- definition_ref: vllm/engine/arg_utils.py:1148
- read_ref: vllm/vllm/config/scheduler.py:123, vllm/vllm/config/vllm.py:1070, vllm/vllm/config/vllm.py:1085
- effect_ref: vllm/vllm/config/vllm.py:1070, vllm/vllm/config/vllm.py:1099, vllm/vllm/v1/core/kv_cache_utils.py:1223
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动或运行 OOM; 缓存不足导致吞吐下降
- value_failure_signals: 启动或运行 OOM; 缓存不足导致吞吐下降
- recommendation: 先保守设置，再渐进放大。
- updated_at: 2026-03-05
