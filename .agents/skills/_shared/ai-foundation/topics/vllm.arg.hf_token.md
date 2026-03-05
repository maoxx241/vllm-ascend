---
topic_id: vllm.arg.hf_token
canonical_term: --hf-token
topic_kind: parameter
---

# --hf-token

## Core

- topic_id: `vllm.arg.hf_token`
- canonical_term: `--hf-token`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `security_auth`
- status/confidence: `needs_manual_review` / `0.76`
- semantics: 控制 API 鉴权和 TLS 安全边界。
- aliases: `--hf-token`, `hf-token`, `hf_token`, `hf token`, `hftoken`, `security_auth`, `security auth`, `security-auth`

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

- definition_ref: vllm/engine/arg_utils.py:700
- read_ref: vllm/vllm/config/model.py:248, vllm/vllm/config/model.py:342, vllm/vllm/config/model.py:495
- effect_ref: vllm/vllm/config/model.py:248, vllm/vllm/config/model.py:342, vllm/vllm/config/model.py:495
- web_refs: 3

## Details/Edge Cases

- failure_modes: 401/403; TLS 握手失败
- value_failure_signals: 401/403; TLS 握手失败
- recommendation: 生产默认开启鉴权并最小化 CORS 白名单。
- updated_at: 2026-03-05
