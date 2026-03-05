---
topic_id: vllm.arg.prefix_caching_hash_algo
canonical_term: --prefix-caching-hash-algo
topic_kind: parameter
---

# --prefix-caching-hash-algo

## Core

- topic_id: `vllm.arg.prefix_caching_hash_algo`
- canonical_term: `--prefix-caching-hash-algo`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `prefix_cache`
- status/confidence: `needs_manual_review` / `0.79`
- semantics: 复用公共前缀缓存，降低 prefill 计算成本。
- aliases: `--prefix-caching-hash-algo`, `prefix-caching-hash-algo`, `prefix_caching_hash_algo`, `prefix caching hash algo`, `prefixcachinghashalgo`, `prefix_cache`, `prefix cache`, `prefix-cache`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `prefix_cache` 查看稳定原理。

## Deployment View

- default_behavior: 默认 sha256。
- value_shape: `enum`
- accepted_values: sha256, sha256_cbor, xxhash, xxhash_cbor
- constraints: xxhash/xxhash_cbor 需要安装可选 xxhash 依赖。; 多租户场景使用非加密哈希存在碰撞与潜在信息泄露风险。
- combo_effects: 与 --enable-prefix-caching 联动；关闭前缀缓存时该值不产生实际效果。

## Development View

- definition_ref: vllm/engine/arg_utils.py:943
- read_ref: vllm/vllm/config/cache.py:78, vllm/vllm/config/cache.py:190, vllm/vllm/engine/arg_utils.py:430
- effect_ref: vllm/vllm/config/cache.py:78, vllm/vllm/config/cache.py:190, vllm/vllm/engine/arg_utils.py:430
- web_refs: 5

## Details/Edge Cases

- failure_modes: 命中率低导致收益不明显; 缓存策略与分块预填充冲突
- value_failure_signals: 缺少 xxhash 依赖时启用 xxhash 系列会失败。
- recommendation: 结合业务前缀分布评估收益，保留回退开关。
- updated_at: 2026-03-05
