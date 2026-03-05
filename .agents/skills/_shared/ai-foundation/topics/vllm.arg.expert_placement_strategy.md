---
topic_id: vllm.arg.expert_placement_strategy
canonical_term: --expert-placement-strategy
topic_kind: parameter
---

# --expert-placement-strategy

## Core

- topic_id: `vllm.arg.expert_placement_strategy`
- canonical_term: `--expert-placement-strategy`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `general_runtime`
- status/confidence: `needs_manual_review` / `0.76`
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `--expert-placement-strategy`, `expert-placement-strategy`, `expert_placement_strategy`, `expert placement strategy`, `expertplacementstrategy`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: 默认 linear。
- value_shape: `enum`
- accepted_values: linear, round_robin
- constraints: 主要在 MoE + expert parallel 场景生效。
- combo_effects: 与 --enable-expert-parallel / --enable-eplb 配合决定最终专家负载分布。

## Development View

- definition_ref: vllm/engine/arg_utils.py:897
- read_ref: vllm/vllm/config/parallel.py:141, vllm/vllm/engine/arg_utils.py:420, vllm/vllm/engine/arg_utils.py:421
- effect_ref: vllm/vllm/model_executor/layers/fused_moe/layer.py:126, vllm/vllm/model_executor/layers/fused_moe/layer.py:131, vllm/vllm/model_executor/layers/fused_moe/layer.py:174
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 非法枚举值会在配置解析阶段报错。
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-05
