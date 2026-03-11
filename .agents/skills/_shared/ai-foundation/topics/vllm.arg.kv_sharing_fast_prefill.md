---
topic_id: vllm.arg.kv_sharing_fast_prefill
canonical_term: --kv-sharing-fast-prefill
topic_kind: parameter
---

# --kv-sharing-fast-prefill

## Core

- topic_id: `vllm.arg.kv_sharing_fast_prefill`
- canonical_term: `--kv-sharing-fast-prefill`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `prefill_decode_disaggregation`
- status/confidence: `aligned` / `0.88`
- source: `code` / source_tags: code
- semantics: 预填充与解码分离部署，优化资源利用和吞吐扩展。
- aliases: `--kv-sharing-fast-prefill`, `kv-sharing-fast-prefill`, `kv_sharing_fast_prefill`, `kv sharing fast prefill`, `kvsharingfastprefill`, `prefill_decode_disaggregation`, `prefill decode disaggregation`, `prefill-decode-disaggregation`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `prefill_decode_disaggregation` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 单机简化部署无法完整覆盖
- combo_effects: N/A

## Development View

- definition_ref: vllm/engine/arg_utils.py:952
- read_ref: vllm/vllm/config/cache.py:152, vllm/vllm/config/cache.py:208, vllm/vllm/config/vllm.py:866
- effect_ref: vllm/vllm/config/vllm.py:866, vllm/vllm/v1/worker/gpu_model_runner.py:648, vllm/vllm/v1/worker/gpu_model_runner.py:1780
- web_refs: 3

## Details/Edge Cases

- failure_modes: connector 超时; P/D 节点路由异常
- value_failure_signals: connector 超时; P/D 节点路由异常
- recommendation: 先验证连接器与地址，再调并行参数。
- updated_at: 2026-03-11
