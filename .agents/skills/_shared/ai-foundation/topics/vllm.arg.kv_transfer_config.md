---
topic_id: vllm.arg.kv_transfer_config
canonical_term: --kv-transfer-config
topic_kind: parameter
---

# --kv-transfer-config

## Core

- topic_id: `vllm.arg.kv_transfer_config`
- canonical_term: `--kv-transfer-config`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `prefill_decode_disaggregation`
- status/confidence: `needs_manual_review` / `0.83`
- source: `code` / source_tags: code
- semantics: 预填充与解码分离部署，优化资源利用和吞吐扩展。
- aliases: `--kv-transfer-config`, `kv-transfer-config`, `kv_transfer_config`, `kv transfer config`, `kvtransferconfig`, `prefill_decode_disaggregation`, `prefill decode disaggregation`, `prefill-decode-disaggregation`

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

- definition_ref: vllm/engine/arg_utils.py:1197
- read_ref: vllm/vllm/config/vllm.py:254, vllm/vllm/config/vllm.py:352, vllm/vllm/config/vllm.py:353
- effect_ref: vllm/vllm/config/vllm.py:352, vllm/vllm/config/vllm.py:564, vllm/vllm/config/vllm.py:1072
- web_refs: 5

## Details/Edge Cases

- failure_modes: connector 超时; P/D 节点路由异常
- value_failure_signals: connector 超时; P/D 节点路由异常
- recommendation: 先验证连接器与地址，再调并行参数。
- updated_at: 2026-03-06
