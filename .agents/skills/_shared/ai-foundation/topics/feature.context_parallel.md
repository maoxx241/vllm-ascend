---
topic_id: feature.context_parallel
canonical_term: context_parallel
topic_kind: feature
---

# Feature: context_parallel

## Core

- topic_id: `feature.context_parallel`
- canonical_term: `context_parallel`
- aliases: `context_parallel`, `context parallel`, `context-parallel`, `上下文并行`, `长上下文并行`, `cp并行`, `cp`, `cp=`, `长序列并行`

## Foundation

- 该特性属于部署/推理配置能力。

## Deployment View

- 部署时应先检查模型/硬件前置条件，再开启。

## Development View

- 开发时应核验定义-读取-生效的完整证据链。

## Details/Edge Cases

- 与参数 topic 通过 `primary_feature` 关联，所有值语义在参数 topic 中展开。
