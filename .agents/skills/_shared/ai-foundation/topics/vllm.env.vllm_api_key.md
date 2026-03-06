---
topic_id: vllm.env.vllm_api_key
canonical_term: VLLM_API_KEY
topic_kind: parameter
---

# VLLM_API_KEY

## Core

- topic_id: `vllm.env.vllm_api_key`
- canonical_term: `VLLM_API_KEY`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `network_serving`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: 控制服务监听、路由和 API 暴露。
- aliases: `VLLM_API_KEY`, `vllm_api_key`, `vllm-api-key`, `vllm api key`, `network_serving`, `network serving`, `network-serving`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `network_serving` 查看稳定原理。

## Deployment View

- default_behavior: 默认 None（不从环境注入鉴权 token）。
- value_shape: `string_secret`
- accepted_values: non-empty token string
- constraints: CLI `--api-key` 优先级高于环境变量。; 密钥应通过安全渠道注入，避免出现在日志或命令历史。
- combo_effects: 与 serve 启动参数共同决定 AuthenticationMiddleware 是否启用。

## Development View

- definition_ref: vllm/envs.py:614
- read_ref: vllm/vllm/entrypoints/openai/api_server.py:239, vllm/vllm/entrypoints/openai/api_server.py:240, vllm/vllm/envs.py:27
- effect_ref: vllm/vllm/entrypoints/openai/api_server.py:240
- web_refs: 2

## Details/Edge Cases

- failure_modes: Address already in use; 健康检查 5xx
- value_failure_signals: 未设置且无 CLI key 时接口可能处于无鉴权状态（依部署策略）。
- recommendation: 固定 host/port 并配套探活。
- updated_at: 2026-03-06
