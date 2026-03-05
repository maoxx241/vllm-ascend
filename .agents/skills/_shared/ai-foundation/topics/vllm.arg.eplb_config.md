---
topic_id: vllm.arg.eplb_config
canonical_term: --eplb-config
topic_kind: parameter
---

# --eplb-config

## Core

- topic_id: `vllm.arg.eplb_config`
- canonical_term: `--eplb-config`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `expert_parallel`
- status/confidence: `needs_manual_review` / `0.79`
- semantics: MoE 专家并行，提升大规模专家模型吞吐。
- aliases: `--eplb-config`, `eplb-config`, `eplb_config`, `eplb config`, `eplbconfig`, `expert_parallel`, `expert parallel`, `expert-parallel`

## Foundation

- EP 面向 MoE 专家路由，Dense 模型没有专家层时不成立。
- 推荐结合 feature: `expert_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 默认使用 EPLBConfig 默认值（window=1000, step_interval=3000, policy=default 等）。
- value_shape: `json_object`
- accepted_values: window_size, step_interval, num_redundant_experts, log_balancedness, log_balancedness_interval, use_async, policy
- constraints: use_async=True 仅支持 policy=default。; log_balancedness=True 时 log_balancedness_interval 必须 > 0。; 当 enable_eplb=False 且 num_redundant_experts!=0 会报错。
- combo_effects: 仅在 --enable-eplb 打开时完整生效。

## Development View

- definition_ref: vllm/engine/arg_utils.py:896
- read_ref: vllm/vllm/config/parallel.py:139, vllm/vllm/config/parallel.py:339, vllm/vllm/config/parallel.py:342
- effect_ref: vllm/vllm/config/parallel.py:339, vllm/vllm/engine/arg_utils.py:605, vllm/vllm/model_executor/models/exaone_moe.py:107
- web_refs: 4

## Details/Edge Cases

- failure_modes: 启动报模型不支持 EP; 专家路由异常
- value_failure_signals: ValueError: Async EPLB is only supported with the default policy.; ValueError: num_redundant_experts ... but EPLB is not enabled
- recommendation: 仅在 MoE profile 启用，并配合 TP/DP 校验。
- updated_at: 2026-03-05
