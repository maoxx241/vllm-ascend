---
topic_id: vllm.arg.trust_remote_code
canonical_term: --trust-remote-code
topic_kind: parameter
---

# --trust-remote-code

## Core

- topic_id: `vllm.arg.trust_remote_code`
- canonical_term: `--trust-remote-code`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `security_auth`
- status/confidence: `aligned` / `0.95`
- source: `code` / source_tags: code
- semantics: 控制 API 鉴权和 TLS 安全边界。
- aliases: `--trust-remote-code`, `trust-remote-code`, `trust_remote_code`, `trust remote code`, `trustremotecode`, `security_auth`, `security auth`, `security-auth`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `security_auth` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 错误证书路径会导致启动失败
- combo_effects: N/A

## Development View

- definition_ref: vllm/engine/arg_utils.py:652
- read_ref: vllm/vllm/benchmarks/serve.py:1571, vllm/vllm/benchmarks/serve.py:1571, vllm/vllm/benchmarks/throughput.py:257
- effect_ref: vllm/vllm/model_executor/models/nemotron_vl.py:430, vllm/vllm/model_executor/models/transformers/utils.py:228, vllm/vllm/tokenizers/hf.py:97
- web_refs: 6

## Details/Edge Cases

- failure_modes: 401/403; TLS 握手失败
- value_failure_signals: 401/403; TLS 握手失败
- recommendation: 生产默认开启鉴权并最小化 CORS 白名单。
- updated_at: 2026-03-06
