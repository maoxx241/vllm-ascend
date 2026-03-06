---
topic_id: vllm_ascend.env.vllm_ascend_fusion_op_transpose_kv_cache_by_block
canonical_term: VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK
topic_kind: parameter
---

# VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK

## Core

- topic_id: `vllm_ascend.env.vllm_ascend_fusion_op_transpose_kv_cache_by_block`
- canonical_term: `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `memory_tuning`
- status/confidence: `aligned` / `0.88`
- source: `code` / source_tags: code_definition, code_reference
- semantics: 控制 KV/权重/中间缓存占用，平衡容量与性能。
- aliases: `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK`, `vllm_ascend_fusion_op_transpose_kv_cache_by_block`, `vllm-ascend-fusion-op-transpose-kv-cache-by-block`, `vllm ascend fusion op transpose kv cache by block`, `memory_tuning`, `memory tuning`, `memory-tuning`

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

- definition_ref: vllm_ascend/envs.py:117, vllm_ascend/envs.py:118
- read_ref: vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py:569, vllm-ascend/vllm_ascend/envs.py:117, vllm-ascend/vllm_ascend/envs.py:118
- effect_ref: vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py:569, vllm-ascend/vllm_ascend/envs.py:117, vllm-ascend/vllm_ascend/envs.py:118
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动或运行 OOM; 缓存不足导致吞吐下降
- value_failure_signals: 启动或运行 OOM; 缓存不足导致吞吐下降
- recommendation: 先保守设置，再渐进放大。
- updated_at: 2026-03-06
