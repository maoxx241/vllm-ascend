---
topic_id: feature.int4_quantization
canonical_term: int4_quantization
topic_kind: feature
---

# Feature: int4_quantization

## Core

- topic_id: `feature.int4_quantization`
- canonical_term: `int4_quantization`
- aliases: `int4_quantization`, `int4 quantization`, `int4-quantization`, `int4量化`, `w4a4`, `4bit量化`, `int4`, `4bit`, `开int4`, `开4bit`

## Foundation

- INT4/W4A4 需要模型工件、内核和平台三方同时支持。

## Deployment View

- 未验证工件必须 hard block，避免线上误启动。

## Development View

- 增加 profile 规则，明确哪些模型可用 INT4。

## Details/Edge Cases

- 与参数 topic 通过 `primary_feature` 关联，所有值语义在参数 topic 中展开。
