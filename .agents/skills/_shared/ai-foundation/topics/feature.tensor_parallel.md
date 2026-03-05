---
topic_id: feature.tensor_parallel
canonical_term: tensor_parallel
topic_kind: feature
---

# Feature: tensor_parallel

## Core

- topic_id: `feature.tensor_parallel`
- canonical_term: `tensor_parallel`
- aliases: `tensor_parallel`, `tensor parallel`, `tensor-parallel`, `张量并行`, `tp并行`, `切tp`, `tp`, `tp=`, `横切并行`

## Foundation

- 该特性属于部署/推理配置能力。

## Deployment View

- 部署时应先检查模型/硬件前置条件，再开启。

## Development View

- 开发时应核验定义-读取-生效的完整证据链。

## Details/Edge Cases

- 与参数 topic 通过 `primary_feature` 关联，所有值语义在参数 topic 中展开。
