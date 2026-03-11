---
topic_id: vllm.arg.enable_sleep_mode
canonical_term: --enable-sleep-mode
topic_kind: parameter
---

# --enable-sleep-mode

## Core

- topic_id: `vllm.arg.enable_sleep_mode`
- canonical_term: `--enable-sleep-mode`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `expert_parallel`
- status/confidence: `aligned` / `0.98`
- source: `code` / source_tags: code
- semantics: 开启 sleep 模式以降低空闲资源占用。
- aliases: `--enable-sleep-mode`, `enable-sleep-mode`, `enable_sleep_mode`, `enable sleep mode`, `enablesleepmode`, `expert_parallel`, `expert parallel`, `expert-parallel`

## Foundation

- EP 面向 MoE 专家路由，Dense 模型没有专家层时不成立。
- 推荐结合 feature: `expert_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: Dense 模型不适用
- combo_effects: N/A

## Development View

- definition_ref: vllm/engine/arg_utils.py:718
- read_ref: vllm/vllm/config/model.py:267, vllm/vllm/config/model.py:472, vllm/vllm/engine/arg_utils.py:555
- effect_ref: vllm/vllm/config/model.py:472, vllm/vllm/engine/arg_utils.py:719, vllm/vllm/v1/worker/gpu_worker.py:202
- web_refs: 5

## Details/Edge Cases

- failure_modes: 启动报模型不支持 EP; 专家路由异常
- value_failure_signals: 启动报模型不支持 EP; 专家路由异常
- recommendation: 仅在 MoE profile 启用，并配合 TP/DP 校验。
- updated_at: 2026-03-11
