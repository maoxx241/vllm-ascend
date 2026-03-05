---
topic_id: feature.expert_parallel
canonical_term: expert_parallel
topic_kind: feature
---

# Feature: expert_parallel

## Core

- topic_id: `feature.expert_parallel`
- canonical_term: `expert_parallel`
- aliases: `expert_parallel`, `expert parallel`, `expert-parallel`, `专家并行`, `ep并行`, `ep`, `ep=`, `moe并行`, `moe`

## Foundation

- EP 面向 MoE 专家路由，Dense 模型没有专家层时不成立。

## Deployment View

- 先判定模型是否 MoE，再决定是否开启 EP。

## Development View

- 在模型画像中固化 has_moe_layers 与专家数量。

## Details/Edge Cases

- 与参数 topic 通过 `primary_feature` 关联，所有值语义在参数 topic 中展开。
