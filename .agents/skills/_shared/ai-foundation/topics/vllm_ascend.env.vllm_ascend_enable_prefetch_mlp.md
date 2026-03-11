---
topic_id: vllm_ascend.env.vllm_ascend_enable_prefetch_mlp
canonical_term: VLLM_ASCEND_ENABLE_PREFETCH_MLP
topic_kind: parameter
---

# VLLM_ASCEND_ENABLE_PREFETCH_MLP

## Core

- topic_id: `vllm_ascend.env.vllm_ascend_enable_prefetch_mlp`
- canonical_term: `VLLM_ASCEND_ENABLE_PREFETCH_MLP`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `weight_prefetch`
- status/confidence: `aligned` / `0.98`
- source: `multi_source` / source_tags: code_definition, code_reference, docs_export
- semantics: 旧版 MLP 预取开关（已在新版本迁移到 additional_config 的 weight_prefetch_config）。
- aliases: `VLLM_ASCEND_ENABLE_PREFETCH_MLP`, `vllm_ascend_enable_prefetch_mlp`, `vllm-ascend-enable-prefetch-mlp`, `vllm ascend enable prefetch mlp`, `weight_prefetch`, `weight prefetch`, `weight-prefetch`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `weight_prefetch` 查看稳定原理。

## Deployment View

- default_behavior: 默认 0（关闭）。
- value_shape: `binary_toggle`
- accepted_values: 0, 1
- constraints: 该变量已标记弃用，后续版本将移除
- combo_effects: 与 VLLM_ASCEND_MLP_GATE_UP_PREFETCH_SIZE / DOWN_PREFETCH_SIZE 联动

## Development View

- definition_ref: docs/source/tutorials/features/suffix_speculative_decoding.md:84, vllm_ascend/ascend_config.py:143, vllm_ascend/envs.py:81
- read_ref: vllm-ascend/vllm_ascend/ascend_config.py:143, vllm-ascend/vllm_ascend/ascend_config.py:151, vllm-ascend/vllm_ascend/ascend_config.py:156
- effect_ref: vllm-ascend/vllm_ascend/ascend_config.py:143, vllm-ascend/vllm_ascend/ascend_config.py:156
- web_refs: 7

## Details/Edge Cases

- failure_modes: 显存 OOM; 收益不稳定
- value_failure_signals: DeprecationWarning: VLLM_ASCEND_ENABLE_PREFETCH_MLP is deprecated
- recommendation: 与 max_model_len/gpu_memory_utilization 联动调优。
- updated_at: 2026-03-11
