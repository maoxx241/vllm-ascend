---
topic_id: vllm_ascend.arg.encoder_dispatch_mode
canonical_term: --encoder-dispatch-mode
topic_kind: parameter
---

# --encoder-dispatch-mode

## Core

- topic_id: `vllm_ascend.arg.encoder_dispatch_mode`
- canonical_term: `--encoder-dispatch-mode`
- kind/scope: `arg` / `vllm_ascend`
- stage: `startup`
- primary_feature: `general_runtime`
- status/confidence: `needs_manual_review` / `0.76`
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `--encoder-dispatch-mode`, `encoder-dispatch-mode`, `encoder_dispatch_mode`, `encoder dispatch mode`, `encoderdispatchmode`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: Encoder dispatch mode: single (one request) or fanout (per-MM-item)
- value_shape: `enum`
- accepted_values: single, fanout
- constraints: 错误组合可能影响稳定性
- combo_effects: N/A

## Development View

- definition_ref: examples/disaggregated_encoder/disagg_epd_proxy.py:717
- read_ref: vllm-ascend/examples/disaggregated_encoder/disagg_epd_proxy.py:216, vllm-ascend/examples/disaggregated_encoder/disagg_epd_proxy.py:735, vllm-ascend/examples/disaggregated_encoder/disagg_epd_proxy.py:735
- effect_ref: vllm-ascend/examples/disaggregated_encoder/disagg_epd_proxy.py:216, vllm-ascend/examples/disaggregated_encoder/disagg_epd_proxy.py:735, vllm-ascend/examples/disaggregated_encoder/disagg_epd_proxy.py:735
- web_refs: 2

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-05
