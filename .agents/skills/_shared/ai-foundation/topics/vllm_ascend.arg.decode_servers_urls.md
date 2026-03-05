---
topic_id: vllm_ascend.arg.decode_servers_urls
canonical_term: --decode-servers-urls
topic_kind: parameter
---

# --decode-servers-urls

## Core

- topic_id: `vllm_ascend.arg.decode_servers_urls`
- canonical_term: `--decode-servers-urls`
- kind/scope: `arg` / `vllm_ascend`
- stage: `startup`
- primary_feature: `prefill_decode_disaggregation`
- status/confidence: `needs_manual_review` / `0.76`
- semantics: 预填充与解码分离部署，优化资源利用和吞吐扩展。
- aliases: `--decode-servers-urls`, `decode-servers-urls`, `decode_servers_urls`, `decode servers urls`, `decodeserversurls`, `prefill_decode_disaggregation`, `prefill decode disaggregation`, `prefill-decode-disaggregation`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `prefill_decode_disaggregation` 查看稳定原理。

## Deployment View

- default_behavior: Comma-separated decode URLs ("http://d1:8005,http://d2:8006")
- value_shape: `free_form`
- accepted_values: string value
- constraints: 单机简化部署无法完整覆盖
- combo_effects: N/A

## Development View

- definition_ref: examples/disaggregated_encoder/disagg_epd_proxy.py:711
- read_ref: vllm-ascend/examples/disaggregated_encoder/disagg_epd_proxy.py:726, vllm-ascend/examples/disaggregated_encoder/disagg_epd_proxy.py:712
- effect_ref: vllm-ascend/examples/disaggregated_encoder/disagg_epd_proxy.py:726
- web_refs: 2

## Details/Edge Cases

- failure_modes: connector 超时; P/D 节点路由异常
- value_failure_signals: connector 超时; P/D 节点路由异常
- recommendation: 先验证连接器与地址，再调并行参数。
- updated_at: 2026-03-05
