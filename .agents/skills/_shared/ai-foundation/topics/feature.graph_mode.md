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

- 图模式通过稳定执行图降低调度抖动，提升吞吐稳定性。

## Deployment View

- 先小流量验证，再放大并发。

## Development View

- 关注图捕获边界、动态 shape 分支、fallback 到 eager 的触发条件。

## Details/Edge Cases

- 与参数 topic 通过 `primary_feature` 关联，所有值语义在参数 topic 中展开。
