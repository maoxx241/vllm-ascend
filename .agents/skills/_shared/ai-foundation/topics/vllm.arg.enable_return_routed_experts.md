---
topic_id: vllm.arg.enable_return_routed_experts
canonical_term: --enable-return-routed-experts
topic_kind: parameter
---

# --enable-return-routed-experts

## Core

- topic_id: `vllm.arg.enable_return_routed_experts`
- canonical_term: `--enable-return-routed-experts`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `expert_parallel`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code
- semantics: MoE 专家并行，提升大规模专家模型吞吐。
- aliases: `--enable-return-routed-experts`, `enable-return-routed-experts`, `enable_return_routed_experts`, `enable return routed experts`, `enablereturnroutedexperts`, `expert_parallel`, `expert parallel`, `expert-parallel`

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

- definition_ref: vllm/engine/arg_utils.py:676
- read_ref: vllm/vllm/config/model.py:199, vllm/vllm/config/vllm.py:1461, vllm/vllm/config/vllm.py:1461
- effect_ref: vllm/vllm/entrypoints/llm.py:169, vllm/vllm/v1/core/sched/scheduler.py:254, vllm/vllm/v1/core/sched/scheduler.py:1513
- web_refs: 4

## Details/Edge Cases

- failure_modes: 启动报模型不支持 EP; 专家路由异常
- value_failure_signals: 启动报模型不支持 EP; 专家路由异常
- recommendation: 仅在 MoE profile 启用，并配合 TP/DP 校验。
- updated_at: 2026-03-06
