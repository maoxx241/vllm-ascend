---
topic_id: vllm_ascend.env.vllm_ascend_enable_nz
canonical_term: VLLM_ASCEND_ENABLE_NZ
topic_kind: parameter
---

# VLLM_ASCEND_ENABLE_NZ

## Core

- topic_id: `vllm_ascend.env.vllm_ascend_enable_nz`
- canonical_term: `VLLM_ASCEND_ENABLE_NZ`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `aligned` / `0.95`
- semantics: 控制 NZ 相关优化路径，部分浮点场景建议关闭或设为特定值。
- aliases: `VLLM_ASCEND_ENABLE_NZ`, `vllm_ascend_enable_nz`, `vllm-ascend-enable-nz`, `vllm ascend enable nz`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `numeric`
- accepted_values: int value
- constraints: 错误组合可能影响稳定性
- combo_effects: N/A

## Development View

- definition_ref: vllm_ascend/batch_invariant.py:80, vllm_ascend/envs.py:101
- read_ref: vllm-ascend/vllm_ascend/batch_invariant.py:80, vllm-ascend/vllm_ascend/envs.py:101, vllm-ascend/vllm_ascend/envs.py:101
- effect_ref: vllm-ascend/vllm_ascend/utils.py:147, vllm-ascend/vllm_ascend/utils.py:152, vllm-ascend/vllm_ascend/worker/worker.py:212
- web_refs: 5

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-05
