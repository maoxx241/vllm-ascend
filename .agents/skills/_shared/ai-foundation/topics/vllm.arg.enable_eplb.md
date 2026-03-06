---
topic_id: vllm.arg.enable_eplb
canonical_term: --enable-eplb
topic_kind: parameter
---

# --enable-eplb

## Core

- topic_id: `vllm.arg.enable_eplb`
- canonical_term: `--enable-eplb`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `expert_parallel`
- status/confidence: `needs_manual_review` / `0.79`
- source: `code` / source_tags: code
- semantics: MoE 专家并行，提升大规模专家模型吞吐。
- aliases: `--enable-eplb`, `enable-eplb`, `enable_eplb`, `enable eplb`, `enableeplb`, `expert_parallel`, `expert parallel`, `expert-parallel`

## Foundation

- EP 面向 MoE 专家路由，Dense 模型没有专家层时不成立。
- 推荐结合 feature: `expert_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 默认 disabled（False）。
- value_shape: `binary_toggle`
- accepted_values: enabled, disabled
- constraints: 仅 CUDA/ROCm 设备支持。; 必须同时 enable_expert_parallel=True。; 要求 TP*DP > 1。
- combo_effects: 与 --eplb-config 联动决定窗口、异步策略与冗余专家策略。

## Development View

- definition_ref: vllm/engine/arg_utils.py:895
- read_ref: vllm/vllm/config/parallel.py:137, vllm/vllm/config/parallel.py:324, vllm/vllm/engine/arg_utils.py:419
- effect_ref: vllm/vllm/config/parallel.py:324, vllm/vllm/engine/arg_utils.py:895, vllm/vllm/model_executor/layers/fused_moe/config.py:884
- web_refs: 4

## Details/Edge Cases

- failure_modes: 启动报模型不支持 EP; 专家路由异常
- value_failure_signals: ValueError: enable_expert_parallel must be True to use EPLB.; ValueError: EPLB requires tensor_parallel_size or data_parallel_size to be greater than 1.
- recommendation: 仅在 MoE profile 启用，并配合 TP/DP 校验。
- updated_at: 2026-03-06
