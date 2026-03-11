---
topic_id: vllm.arg.enable_expert_parallel
canonical_term: --enable-expert-parallel
topic_kind: parameter
---

# --enable-expert-parallel

## Core

- topic_id: `vllm.arg.enable_expert_parallel`
- canonical_term: `--enable-expert-parallel`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `expert_parallel`
- status/confidence: `aligned` / `0.98`
- source: `code` / source_tags: code
- semantics: 开启 MoE 专家并行，仅对 MoE 模型有效。
- aliases: `--enable-expert-parallel`, `enable-expert-parallel`, `enable_expert_parallel`, `enable expert parallel`, `enableexpertparallel`, `expert_parallel`, `expert parallel`, `expert-parallel`

## Foundation

- EP 面向 MoE 专家路由，Dense 模型没有专家层时不成立。
- 推荐结合 feature: `expert_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 默认 disabled（False）。
- value_shape: `binary_toggle`
- accepted_values: enabled, disabled
- constraints: 仅 MoE 模型可开启；dense 模型会报错。
- combo_effects: 与 --all2all-backend、--enable-eplb、TP/DP 参数强耦合。; qwen3-32b-w8a8 profile 下与 EP 组合应硬阻断（见 combo rule）。

## Development View

- definition_ref: vllm/engine/arg_utils.py:869
- read_ref: vllm/vllm/config/model.py:1060, vllm/vllm/config/parallel.py:135, vllm/vllm/config/parallel.py:330
- effect_ref: vllm/vllm/config/model.py:1060, vllm/vllm/config/parallel.py:330, vllm/vllm/config/parallel.py:331
- web_refs: 6

## Details/Edge Cases

- failure_modes: 启动报模型不支持 EP; 专家路由异常
- value_failure_signals: ValueError: Number of experts in the model must be greater than 0 when expert parallelism is enabled.
- recommendation: 仅在 MoE profile 启用，并配合 TP/DP 校验。
- updated_at: 2026-03-11
