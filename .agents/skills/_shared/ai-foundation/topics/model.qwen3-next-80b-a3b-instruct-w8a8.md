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
- base_model: `N/A`

## Foundation

- 模型画像用于配置可行性推导，不参与业务逻辑改写。
- Dense 模型不适用 EP；MoE 模型可进一步评估 EP；资源建议采用 recommended/validated 双口径。

## Deployment View

- 部署前先做 profile 校验：量化工件支持、并行能力边界、resource_guidance（recommended/validated/boot_min）。
- 资源不足时输出 advisory + fallback，不把建议卡数当成硬门槛降级。

## Development View

- evidence_refs: .agents/skills/_shared/vllm-ascend-core/concepts/model-feature-compatibility-matrix.md, docs/source/tutorials/models/Qwen3-Next.md
- resource_guidance.recommended: data_parallel: >= 8 (80B MoE 常用多副本扩展吞吐。); context_parallel: >= 8 (长上下文 CP 建议从高卡场景评估。)
- resource_guidance.validated: tensor_parallel: >= 4 (官方教程示例覆盖 TP4。); expert_parallel: >= 8 (EP 需结合 MoE 路由与更高通信预算。)
- resource_guidance.boot_min: >= 4 (基础启动可从 4 卡开始。)

## Details/Edge Cases

- supported_variants: ['w8a8']
- architecture_family: qwen3_next
- variant_scope.notes: N/A
- resource_guidance.evidence_refs: docs/source/tutorials/models/Qwen3-Next.md
