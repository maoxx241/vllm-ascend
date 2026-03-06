---
topic_id: feature.graph_mode
canonical_term: graph_mode
topic_kind: feature
---

# Feature: graph_mode

## Core

- topic_id: `feature.graph_mode`
- canonical_term: `graph_mode`
- aliases: `graph_mode`, `graph mode`, `graph-mode`, `图模式`, `开图`, `全图`, `图加速`, `cudagraph`, `full decode`, `抓图`

## Foundation

- Ascend 图模式由 ACLGraph 与 Xlite 图配置共同作用，FULL_DECODE_ONLY 常用于先稳态加速 decode。

## Deployment View

- 优先用 --compilation-config {'cudagraph_mode':'FULL_DECODE_ONLY'} 做灰度，必要时用 --enforce-eager 回退。

## Development View

- 关注 full_mode/xlite 约束、block_size 要求、动态 shape 分支与 eager fallback 触发条件。

## Details/Edge Cases

- 与参数 topic 通过 `primary_feature` 关联，所有值语义在参数 topic 中展开。
