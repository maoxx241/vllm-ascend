---
topic_id: vllm_ascend.arg.sleep_mode_level
canonical_term: --sleep-mode-level
topic_kind: parameter
---

# --sleep-mode-level

## Core

- topic_id: `vllm_ascend.arg.sleep_mode_level`
- canonical_term: `--sleep-mode-level`
- kind/scope: `arg` / `vllm_ascend`
- stage: `startup`
- primary_feature: `expert_parallel`
- status/confidence: `needs_manual_review` / `0.79`
- source: `code` / source_tags: code
- semantics: MoE 专家并行，提升大规模专家模型吞吐。
- aliases: `--sleep-mode-level`, `sleep-mode-level`, `sleep_mode_level`, `sleep mode level`, `sleepmodelevel`, `expert_parallel`, `expert parallel`, `expert-parallel`

## Foundation

- EP 面向 MoE 专家路由，Dense 模型没有专家层时不成立。
- 推荐结合 feature: `expert_parallel` 查看稳定原理。

## Deployment View

- default_behavior: Sleep mode level: 1 or 2. This example of level 2 is only supported for dense model.
- value_shape: `enum`
- accepted_values: 1, 2
- constraints: Dense 模型不适用; This example of level 2 is only supported for dense model.
- combo_effects: N/A

## Development View

- definition_ref: examples/offline_external_launcher.py:135
- read_ref: vllm-ascend/examples/offline_external_launcher.py:171, vllm-ascend/examples/offline_external_launcher.py:213, vllm-ascend/examples/offline_external_launcher.py:221
- effect_ref: vllm-ascend/examples/offline_external_launcher.py:221
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动报模型不支持 EP; 专家路由异常
- value_failure_signals: 启动报模型不支持 EP; 专家路由异常
- recommendation: 仅在 MoE profile 启用，并配合 TP/DP 校验。
- updated_at: 2026-03-06
