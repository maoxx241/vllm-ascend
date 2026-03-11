---
topic_id: vllm_ascend.env.ascend_rt_visible_devices
canonical_term: ASCEND_RT_VISIBLE_DEVICES
topic_kind: parameter
---

# ASCEND_RT_VISIBLE_DEVICES

## Core

- topic_id: `vllm_ascend.env.ascend_rt_visible_devices`
- canonical_term: `ASCEND_RT_VISIBLE_DEVICES`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `upstream_delta` / `0.75`
- source: `multi_source` / source_tags: code_reference, docs_export
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `ASCEND_RT_VISIBLE_DEVICES`, `ascend_rt_visible_devices`, `ascend-rt-visible-devices`, `ascend rt visible devices`, `general_runtime`, `general runtime`, `general-runtime`

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

- definition_ref: docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:272, docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:331, docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:390
- read_ref: vllm-ascend/vllm_ascend/cpu_binding.py:16, vllm-ascend/vllm_ascend/cpu_binding.py:16, vllm-ascend/vllm_ascend/cpu_binding.py:90
- effect_ref: vllm-ascend/vllm_ascend/cpu_binding.py:90
- web_refs: 4

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-11
