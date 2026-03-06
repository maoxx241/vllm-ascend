---
topic_id: vllm_ascend.arg.prefill_servers_urls
canonical_term: --prefill-servers-urls
topic_kind: parameter
---

# --prefill-servers-urls

## Core

- topic_id: `vllm_ascend.arg.prefill_servers_urls`
- canonical_term: `--prefill-servers-urls`
- kind/scope: `arg` / `vllm_ascend`
- stage: `startup`
- primary_feature: `prefill_decode_disaggregation`
- status/confidence: `needs_manual_review` / `0.76`
- source: `code` / source_tags: code
- semantics: 预填充与解码分离部署，优化资源利用和吞吐扩展。
- aliases: `--prefill-servers-urls`, `prefill-servers-urls`, `prefill_servers_urls`, `prefill servers urls`, `prefillserversurls`, `prefill_decode_disaggregation`, `prefill decode disaggregation`, `prefill-decode-disaggregation`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `prefill_decode_disaggregation` 查看稳定原理。

## Deployment View

- default_behavior: Comma-separated prefill URLs ("http://p1:8003,http://p2:8004") to enable E->P->D, set "disable" or "none" to enable E->PD
- value_shape: `free_form`
- accepted_values: http://p1:8003,http://p2:8004, disable, none
- constraints: 单机简化部署无法完整覆盖
- combo_effects: N/A

## Development View

- definition_ref: examples/disaggregated_encoder/disagg_epd_proxy.py:705
- read_ref: vllm-ascend/examples/disaggregated_encoder/disagg_epd_proxy.py:728, vllm-ascend/examples/disaggregated_encoder/disagg_epd_proxy.py:732, vllm-ascend/examples/disaggregated_encoder/disagg_epd_proxy.py:706
- effect_ref: vllm-ascend/examples/disaggregated_encoder/disagg_epd_proxy.py:728, vllm-ascend/examples/disaggregated_encoder/disagg_epd_proxy.py:732
- web_refs: 2

## Details/Edge Cases

- failure_modes: connector 超时; P/D 节点路由异常
- value_failure_signals: connector 超时; P/D 节点路由异常
- recommendation: 先验证连接器与地址，再调并行参数。
- updated_at: 2026-03-06
