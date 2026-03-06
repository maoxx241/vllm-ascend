---
topic_id: vllm_ascend.arg.enable_sleep_mode
canonical_term: --enable-sleep-mode
topic_kind: parameter
---

# --enable-sleep-mode

## Core

- topic_id: `vllm_ascend.arg.enable_sleep_mode`
- canonical_term: `--enable-sleep-mode`
- kind/scope: `arg` / `vllm_ascend`
- stage: `startup`
- primary_feature: `expert_parallel`
- status/confidence: `needs_manual_review` / `0.86`
- source: `code` / source_tags: code
- semantics: 开启 sleep 模式以降低空闲资源占用。
- aliases: `--enable-sleep-mode`, `enable-sleep-mode`, `enable_sleep_mode`, `enable sleep mode`, `enablesleepmode`, `expert_parallel`, `expert parallel`, `expert-parallel`

## Foundation

- EP 面向 MoE 专家路由，Dense 模型没有专家层时不成立。
- 推荐结合 feature: `expert_parallel` 查看稳定原理。

## Deployment View

- default_behavior: Enable sleep mode for the engine.
- value_shape: `binary_or_auto`
- accepted_values: enabled, disabled, unset(auto)
- constraints: Dense 模型不适用
- combo_effects: N/A

## Development View

- definition_ref: examples/offline_external_launcher.py:125, examples/offline_weight_load.py:134
- read_ref: vllm/vllm/config/model.py:271, vllm/vllm/config/model.py:469, vllm/vllm/engine/arg_utils.py:555
- effect_ref: vllm/vllm/config/model.py:469, vllm/vllm/engine/arg_utils.py:720, vllm/vllm/v1/worker/gpu_worker.py:171
- web_refs: 4

## Details/Edge Cases

- failure_modes: 启动报模型不支持 EP; 专家路由异常
- value_failure_signals: 启动报模型不支持 EP; 专家路由异常
- recommendation: 仅在 MoE profile 启用，并配合 TP/DP 校验。
- updated_at: 2026-03-06
