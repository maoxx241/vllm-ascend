---
topic_id: vllm.arg.enable_prefix_caching
canonical_term: --enable-prefix-caching
topic_kind: parameter
---

# --enable-prefix-caching

## Core

- topic_id: `vllm.arg.enable_prefix_caching`
- canonical_term: `--enable-prefix-caching`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `prefix_cache`
- status/confidence: `needs_manual_review` / `0.86`
- source: `code` / source_tags: code
- semantics: 启用前缀缓存，加速重复前缀请求的 prefill 阶段。
- aliases: `--enable-prefix-caching`, `enable-prefix-caching`, `enable_prefix_caching`, `enable prefix caching`, `enableprefixcaching`, `prefix_cache`, `prefix cache`, `prefix-cache`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `prefix_cache` 查看稳定原理。

## Deployment View

- default_behavior: EngineArgs 未显式设置时会使用模型/后端默认值；CacheConfig 默认值为 True。
- value_shape: `binary_or_auto`
- accepted_values: enabled, disabled, unset(auto)
- constraints: 在 pooling 等不官方支持场景强行开启会产生风险告警。; 在 POWER/S390X/RISC-V CPU 上会被强制关闭。; 若设置 mamba-block-size，需要开启前缀缓存。
- combo_effects: 与 mamba_cache_mode/mamba_block_size 联动。; 与 prefix_caching_hash_algo 联动决定命中键生成方式。

## Development View

- definition_ref: vllm/engine/arg_utils.py:936
- read_ref: vllm/vllm/benchmarks/latency.py:77, vllm/vllm/config/cache.py:76, vllm/vllm/config/cache.py:189
- effect_ref: vllm/vllm/config/vllm.py:1482, vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:264, vllm/vllm/engine/arg_utils.py:1967
- web_refs: 6

## Details/Edge Cases

- failure_modes: 命中率低导致收益不明显; 缓存策略与分块预填充冲突
- value_failure_signals: warning: model does not officially support prefix caching; ValueError: --mamba-block-size can only be set with --enable-prefix-caching
- recommendation: 结合业务前缀分布评估收益，保留回退开关。
- updated_at: 2026-03-06
