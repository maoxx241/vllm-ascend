---
topic_id: vllm.env.vllm_allow_long_max_model_len
canonical_term: VLLM_ALLOW_LONG_MAX_MODEL_LEN
topic_kind: parameter
---

# VLLM_ALLOW_LONG_MAX_MODEL_LEN

## Core

- topic_id: `vllm.env.vllm_allow_long_max_model_len`
- canonical_term: `VLLM_ALLOW_LONG_MAX_MODEL_LEN`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `memory_tuning`
- status/confidence: `aligned` / `0.98`
- semantics: 控制 KV/权重/中间缓存占用，平衡容量与性能。
- aliases: `VLLM_ALLOW_LONG_MAX_MODEL_LEN`, `vllm_allow_long_max_model_len`, `vllm-allow-long-max-model-len`, `vllm allow long max model len`, `memory_tuning`, `memory tuning`, `memory-tuning`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `memory_tuning` 查看稳定原理。

## Deployment View

- default_behavior: 默认 0（不允许超过模型推导上限的 max_model_len）。
- value_shape: `binary_toggle`
- accepted_values: 0, 1
- constraints: 仅在用户 max_model_len > derived_max_model_len 时生效。; 官方明确提示该开关需极度谨慎使用（RoPE 可能 NaN，绝对位置编码可能 OOB）。
- combo_effects: 与 --max-model-len 强耦合；不开该开关时超长配置会报错。

## Development View

- definition_ref: vllm/envs.py:819
- read_ref: vllm/vllm/config/model.py:2017, vllm/vllm/config/model.py:2024, vllm/vllm/config/model.py:2029
- effect_ref: vllm/vllm/config/model.py:2024, vllm/vllm/envs.py:815, vllm/vllm/envs.py:818
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动或运行 OOM; 缓存不足导致吞吐下降
- value_failure_signals: ValueError: User-specified max_model_len ... To allow overriding this maximum, set VLLM_ALLOW_LONG_MAX_MODEL_LEN=1; warning: positions exceeding derived_max_model_len may lead to NaN/OOB
- recommendation: 先保守设置，再渐进放大。
- updated_at: 2026-03-05
