---
topic_id: vllm_ascend.env.vllm_ascend_enable_topk_optimize
canonical_term: VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE
topic_kind: parameter
---

# VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE

## Core

- topic_id: `vllm_ascend.env.vllm_ascend_enable_topk_optimize`
- canonical_term: `VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `upstream_delta` / `0.48`
- source: `tests_yaml` / source_tags: tests_yaml
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE`, `vllm_ascend_enable_topk_optimize`, `vllm-ascend-enable-topk-optimize`, `vllm ascend enable topk optimize`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 错误组合可能影响稳定性
- combo_effects: N/A

## Development View

- definition_ref: tests/e2e/nightly/single_node/models/configs/Qwen3-32B-Int8-A3-Feature-Stack3.yaml:13
- read_ref: N/A
- effect_ref: N/A
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-06
