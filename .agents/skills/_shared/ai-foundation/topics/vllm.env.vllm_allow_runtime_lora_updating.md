---
topic_id: vllm.env.vllm_allow_runtime_lora_updating
canonical_term: VLLM_ALLOW_RUNTIME_LORA_UPDATING
topic_kind: parameter
---

# VLLM_ALLOW_RUNTIME_LORA_UPDATING

## Core

- topic_id: `vllm.env.vllm_allow_runtime_lora_updating`
- canonical_term: `VLLM_ALLOW_RUNTIME_LORA_UPDATING`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `lora`
- status/confidence: `needs_manual_review` / `0.79`
- source: `code` / source_tags: code_definition
- semantics: 开启 LoRA 适配器加载与路由。
- aliases: `VLLM_ALLOW_RUNTIME_LORA_UPDATING`, `vllm_allow_runtime_lora_updating`, `vllm-allow-runtime-lora-updating`, `vllm allow runtime lora updating`, `lora`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `lora` 查看稳定原理。

## Deployment View

- default_behavior: 默认 0（关闭运行时 LoRA 动态加载/卸载）。
- value_shape: `binary_toggle`
- accepted_values: 0, 1
- constraints: api_server_count > 1 时不允许开启，会直接报错。; 官方警告该能力应仅用于本地开发场景。
- combo_effects: 与 LoRA resolver 相关环境变量联动（插件与仓库解析）。

## Development View

- definition_ref: vllm/envs.py:861
- read_ref: vllm/vllm/entrypoints/cli/serve.py:234, vllm/vllm/entrypoints/cli/serve.py:236, vllm/vllm/entrypoints/openai/engine/serving.py:752
- effect_ref: vllm/vllm/entrypoints/cli/serve.py:234, vllm/vllm/entrypoints/serve/lora/api_router.py:27
- web_refs: 4

## Details/Edge Cases

- failure_modes: LoRA 模块加载报错; 输出异常
- value_failure_signals: ValueError: VLLM_ALLOW_RUNTIME_LORA_UPDATING cannot be used with api_server_count > 1; warning: LoRA dynamic loading & unloading is enabled ... ONLY be used for local development
- recommendation: 先离线验证 LoRA 工件，再接入在线服务。
- updated_at: 2026-03-06
