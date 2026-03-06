---
topic_id: vllm_ascend.env.ascend_custom_opp_path
canonical_term: ASCEND_CUSTOM_OPP_PATH
topic_kind: parameter
---

# ASCEND_CUSTOM_OPP_PATH

## Core

- topic_id: `vllm_ascend.env.ascend_custom_opp_path`
- canonical_term: `ASCEND_CUSTOM_OPP_PATH`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `upstream_delta` / `0.68`
- source: `code` / source_tags: code_reference
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `ASCEND_CUSTOM_OPP_PATH`, `ascend_custom_opp_path`, `ascend-custom-opp-path`, `ascend custom opp path`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 错误组合可能影响稳定性
- combo_effects: N/A

## Development View

- definition_ref: vllm_ascend/platform.py:471, vllm_ascend/platform.py:473
- read_ref: vllm-ascend/vllm_ascend/platform.py:469, vllm-ascend/vllm_ascend/platform.py:471, vllm-ascend/vllm_ascend/platform.py:473
- effect_ref: vllm-ascend/vllm_ascend/platform.py:469, vllm-ascend/vllm_ascend/platform.py:471, vllm-ascend/vllm_ascend/platform.py:473
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-06
