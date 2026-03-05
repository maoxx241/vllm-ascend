---
topic_id: vllm.arg.speculative_config
canonical_term: --speculative-config
topic_kind: parameter
---

# --speculative-config

## Core

- topic_id: `vllm.arg.speculative_config`
- canonical_term: `--speculative-config`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `speculative_decode`
- status/confidence: `needs_manual_review` / `0.86`
- semantics: 配置投机解码策略（如 mtp），用于降低解码延迟。
- aliases: `--speculative-config`, `speculative-config`, `speculative_config`, `speculative config`, `speculativeconfig`, `speculative_decode`, `speculative decode`, `speculative-decode`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `speculative_decode` 查看稳定原理。

## Deployment View

- default_behavior: 默认 None（关闭 speculative decoding）。
- value_shape: `json_object`
- accepted_values: method, model, num_speculative_tokens, draft_tensor_parallel_size, quantization, disable_by_batch_size, disable_padded_drafter_batch, parallel_drafting
- constraints: num_speculative_tokens 必须 > 0。; speculative_config 内不允许 tensor_parallel_size 字段，需使用 draft_tensor_parallel_size。; draft_tensor_parallel_size 仅允许 1 或 target TP。
- combo_effects: 与 --async-scheduling 联动：仅 EAGLE/MTP/draft_model 且 disable_padded_drafter_batch=False 才兼容。; 与 cudagraph size 计算联动：num_speculative_tokens 会影响 decode_query_len 与 graph size 上限。

## Development View

- definition_ref: vllm/engine/arg_utils.py:1194
- read_ref: vllm/vllm/config/speculative.py:661, vllm/vllm/config/vllm.py:231, vllm/vllm/config/vllm.py:335
- effect_ref: vllm/vllm/config/vllm.py:335, vllm/vllm/config/vllm.py:622, vllm/vllm/config/vllm.py:631
- web_refs: 6

## Details/Edge Cases

- failure_modes: 服务启动后推理错误; 吞吐波动
- value_failure_signals: ValueError: num_speculative_tokens must be provided ...; ValueError: 'tensor_parallel_size' is not a valid argument in speculative_config; ValueError: async scheduling is only supported with EAGLE/MTP/Draft ...
- recommendation: 先用小 token 数验证，再逐步增加并发。
- updated_at: 2026-03-05
