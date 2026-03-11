---
topic_id: vllm.arg.dbo_decode_token_threshold
canonical_term: --dbo-decode-token-threshold
topic_kind: parameter
---

# --dbo-decode-token-threshold

## Core

- topic_id: `vllm.arg.dbo_decode_token_threshold`
- canonical_term: `--dbo-decode-token-threshold`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `prefill_decode_disaggregation`
- status/confidence: `aligned` / `0.88`
- source: `code` / source_tags: code
- semantics: 预填充与解码分离部署，优化资源利用和吞吐扩展。
- aliases: `--dbo-decode-token-threshold`, `dbo-decode-token-threshold`, `dbo_decode_token_threshold`, `dbo decode token threshold`, `dbodecodetokenthreshold`, `prefill_decode_disaggregation`, `prefill decode disaggregation`, `prefill-decode-disaggregation`

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

- definition_ref: vllm/engine/arg_utils.py:882
- read_ref: vllm/vllm/config/parallel.py:174, vllm/vllm/engine/arg_utils.py:413, vllm/vllm/engine/arg_utils.py:413
- effect_ref: vllm/vllm/v1/worker/ubatch_utils.py:44
- web_refs: 3

## Details/Edge Cases

- failure_modes: connector 超时; P/D 节点路由异常
- value_failure_signals: connector 超时; P/D 节点路由异常
- recommendation: 先验证连接器与地址，再调并行参数。
- updated_at: 2026-03-11
