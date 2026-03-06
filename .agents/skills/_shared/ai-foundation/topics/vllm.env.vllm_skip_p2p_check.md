---
topic_id: vllm.env.vllm_skip_p2p_check
canonical_term: VLLM_SKIP_P2P_CHECK
topic_kind: parameter
---

# VLLM_SKIP_P2P_CHECK

## Core

- topic_id: `vllm.env.vllm_skip_p2p_check`
- canonical_term: `VLLM_SKIP_P2P_CHECK`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `needs_manual_review` / `0.79`
- source: `code` / source_tags: code_definition
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `VLLM_SKIP_P2P_CHECK`, `vllm_skip_p2p_check`, `vllm-skip-p2p-check`, `vllm skip p2p check`, `general_runtime`, `general runtime`, `general-runtime`

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

- definition_ref: vllm/envs.py:871
- read_ref: vllm/vllm/distributed/device_communicators/custom_all_reduce.py:36, vllm/vllm/envs.py:97, vllm/vllm/envs.py:868
- effect_ref: vllm/vllm/distributed/device_communicators/custom_all_reduce.py:36, vllm/vllm/envs.py:868
- web_refs: 2

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-06
