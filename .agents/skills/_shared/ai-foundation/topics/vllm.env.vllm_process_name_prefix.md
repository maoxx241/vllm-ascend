---
topic_id: vllm.env.vllm_process_name_prefix
canonical_term: VLLM_PROCESS_NAME_PREFIX
topic_kind: parameter
---

# VLLM_PROCESS_NAME_PREFIX

## Core

- topic_id: `vllm.env.vllm_process_name_prefix`
- canonical_term: `VLLM_PROCESS_NAME_PREFIX`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `prefix_cache`
- status/confidence: `needs_manual_review` / `0.79`
- source: `code` / source_tags: code_definition
- semantics: 复用公共前缀缓存，降低 prefill 计算成本。
- aliases: `VLLM_PROCESS_NAME_PREFIX`, `vllm_process_name_prefix`, `vllm-process-name-prefix`, `vllm process name prefix`, `prefix_cache`, `prefix cache`, `prefix-cache`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `prefix_cache` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 部分调度组合性能可能下降
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:1378
- read_ref: vllm/vllm/envs.py:1378, vllm/vllm/envs.py:1378, vllm/vllm/utils/system_utils.py:166
- effect_ref: vllm/vllm/envs.py:1378, vllm/vllm/envs.py:1378, vllm/vllm/utils/system_utils.py:166
- web_refs: 4

## Details/Edge Cases

- failure_modes: 命中率低导致收益不明显; 缓存策略与分块预填充冲突
- value_failure_signals: 命中率低导致收益不明显; 缓存策略与分块预填充冲突
- recommendation: 结合业务前缀分布评估收益，保留回退开关。
- updated_at: 2026-03-06
