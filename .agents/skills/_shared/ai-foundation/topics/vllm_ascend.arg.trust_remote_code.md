---
topic_id: vllm_ascend.arg.trust_remote_code
canonical_term: --trust-remote-code
topic_kind: parameter
---

# --trust-remote-code

## Core

- topic_id: `vllm_ascend.arg.trust_remote_code`
- canonical_term: `--trust-remote-code`
- kind/scope: `arg` / `vllm_ascend`
- stage: `startup`
- primary_feature: `security_auth`
- status/confidence: `needs_manual_review` / `0.83`
- semantics: 控制 API 鉴权和 TLS 安全边界。
- aliases: `--trust-remote-code`, `trust-remote-code`, `trust_remote_code`, `trust remote code`, `trustremotecode`, `security_auth`, `security auth`, `security-auth`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `security_auth` 查看稳定原理。

## Deployment View

- default_behavior: Trust remote code.
- value_shape: `binary_or_auto`
- accepted_values: enabled, disabled, unset(auto)
- constraints: 错误证书路径会导致启动失败
- combo_effects: N/A

## Development View

- definition_ref: examples/offline_data_parallel.py:88, examples/offline_external_launcher.py:121, examples/offline_weight_load.py:130
- read_ref: vllm/vllm/benchmarks/serve.py:1571, vllm/vllm/benchmarks/serve.py:1571, vllm/vllm/benchmarks/throughput.py:257
- effect_ref: vllm/vllm/model_executor/models/nemotron_vl.py:430, vllm/vllm/model_executor/models/transformers/utils.py:228, vllm/vllm/tokenizers/hf.py:97
- web_refs: 5

## Details/Edge Cases

- failure_modes: 401/403; TLS 握手失败
- value_failure_signals: 401/403; TLS 握手失败
- recommendation: 生产默认开启鉴权并最小化 CORS 白名单。
- updated_at: 2026-03-05
