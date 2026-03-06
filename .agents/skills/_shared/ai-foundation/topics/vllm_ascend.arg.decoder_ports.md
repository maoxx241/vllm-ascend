---
topic_id: vllm_ascend.arg.decoder_ports
canonical_term: --decoder-ports
topic_kind: parameter
---

# --decoder-ports

## Core

- topic_id: `vllm_ascend.arg.decoder_ports`
- canonical_term: `--decoder-ports`
- kind/scope: `arg` / `vllm_ascend`
- stage: `startup`
- primary_feature: `prefill_decode_disaggregation`
- status/confidence: `needs_manual_review` / `0.83`
- source: `code` / source_tags: code
- semantics: 预填充与解码分离部署，优化资源利用和吞吐扩展。
- aliases: `--decoder-ports`, `decoder-ports`, `decoder_ports`, `decoder ports`, `decoderports`, `prefill_decode_disaggregation`, `prefill decode disaggregation`, `prefill-decode-disaggregation`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `prefill_decode_disaggregation` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `list`
- accepted_values: list value
- constraints: 单机简化部署无法完整覆盖
- combo_effects: N/A

## Development View

- definition_ref: examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py:265, examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:513
- read_ref: vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py:273, vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py:276, vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:530
- effect_ref: vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py:273, vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:530
- web_refs: 5

## Details/Edge Cases

- failure_modes: connector 超时; P/D 节点路由异常
- value_failure_signals: connector 超时; P/D 节点路由异常
- recommendation: 先验证连接器与地址，再调并行参数。
- updated_at: 2026-03-06
