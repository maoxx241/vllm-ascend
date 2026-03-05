---
topic_id: feature.quantization
canonical_term: quantization
topic_kind: feature
---

# Feature: quantization

## Core

- topic_id: `feature.quantization`
- canonical_term: `quantization`
- aliases: `quantization`, `量化`, `开量化`, `int8量化`, `w8a8`, `int8`, `压模型`, `压权重`

## Foundation

- 量化通过低比特权重/激活表示降低显存和带宽开销，常以精度换吞吐。

## Deployment View

- 先确认模型工件支持，再配置 quantization + dtype + 并行组合。

## Development View

- 核验量化后端分支、算子覆盖率、降级路径和精度监控。

## Details/Edge Cases

- 与参数 topic 通过 `primary_feature` 关联，所有值语义在参数 topic 中展开。
