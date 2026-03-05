---
topic_id: model.qwen3-next-80b-a3b-instruct-w8a8
canonical_term: qwen3-next-80b-a3b-instruct-w8a8
topic_kind: model_profile
---

# Model Profile: qwen3-next-80b-a3b-instruct-w8a8

## Core

- topic_id: `model.qwen3-next-80b-a3b-instruct-w8a8`
- canonical_term: `qwen3-next-80b-a3b-instruct-w8a8`
- has_moe_layers: `True`
- num_experts: `80`
- fixed_weight_format: `w8a8`

## Foundation

- 模型画像用于配置可行性推导，不参与业务逻辑改写。
- Dense 模型不适用 EP；MoE 模型可进一步评估 EP。

## Deployment View

- 部署前先做 profile 校验：量化工件支持、并行能力边界、最小卡数。
- 不满足条件时返回 hard_block/warning，并附 fallback。

## Development View

- evidence_refs: .agents/skills/_shared/vllm-ascend-core/concepts/model-feature-compatibility-matrix.md, docs/source/tutorials/models/Qwen3-Next.md
- feature_min_npu_count: {'data_parallel': 8, 'context_parallel': 8}

## Details/Edge Cases

- supported_variants: ['w8a8']
- architecture_family: qwen3_next
