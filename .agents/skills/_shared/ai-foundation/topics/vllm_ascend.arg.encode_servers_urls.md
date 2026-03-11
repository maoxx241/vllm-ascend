---
topic_id: vllm_ascend.arg.encode_servers_urls
canonical_term: --encode-servers-urls
topic_kind: parameter
---

# --encode-servers-urls

## Core

- topic_id: `vllm_ascend.arg.encode_servers_urls`
- canonical_term: `--encode-servers-urls`
- kind/scope: `arg` / `vllm_ascend`
- stage: `startup`
- primary_feature: `general_runtime`
- status/confidence: `needs_manual_review` / `0.76`
- source: `code` / source_tags: code
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `--encode-servers-urls`, `encode-servers-urls`, `encode_servers_urls`, `encode servers urls`, `encodeserversurls`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: Comma-separated encode URLs ("http://e1:8001,http://e2:8001")
- value_shape: `free_form`
- accepted_values: string value
- constraints: 错误组合可能影响稳定性
- combo_effects: N/A

## Development View

- definition_ref: examples/disaggregated_encoder/disagg_epd_proxy.py:700
- read_ref: vllm-ascend/examples/disaggregated_encoder/disagg_epd_proxy.py:725, vllm-ascend/examples/disaggregated_encoder/disagg_epd_proxy.py:701
- effect_ref: vllm-ascend/examples/disaggregated_encoder/disagg_epd_proxy.py:725
- web_refs: 2

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-11
