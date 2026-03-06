---
topic_id: model.qwen3-32b-w8a8
canonical_term: qwen3-32b-w8a8
topic_kind: model_profile
---

# Model Profile: qwen3-32b-w8a8

## Core

- topic_id: `model.qwen3-32b-w8a8`
- canonical_term: `qwen3-32b-w8a8`
- has_moe_layers: `False`
- num_experts: `0`
- fixed_weight_format: `w8a8`
- base_model: `qwen3-32b`

## Foundation

- 模型画像用于配置可行性推导，不参与业务逻辑改写。
- Dense 模型不适用 EP；MoE 模型可进一步评估 EP；资源建议采用 recommended/validated 双口径。

## Deployment View

- 部署前先做 profile 校验：量化工件支持、并行能力边界、resource_guidance（recommended/validated/boot_min）。
- 资源不足时输出 advisory + fallback，不把建议卡数当成硬门槛降级。

## Development View

- evidence_refs: .agents/skills/_shared/vllm-ascend-core/concepts/model-feature-compatibility-matrix.md, docs/source/tutorials/models/Qwen3-Dense.md
- resource_guidance.recommended: tensor_parallel: >= 4 (单机 TP4 是常见起点。); data_parallel: >= 8 (DP 常用于吞吐扩展，建议 8 卡起评估收益。); context_parallel: >= 8 (CP 主要面向长上下文和高并发场景。)
- resource_guidance.validated: tensor_parallel: >= 4 (Qwen3-Dense 教程与 nightly 32B 配置均覆盖 TP4。); graph_mode: >= 4 (FULL_DECODE_ONLY 在 4 卡配置已有验证样例。)
- resource_guidance.boot_min: >= 4 (当前知识库保证 4 卡 TP4 启动基线；更低卡数需专项验证。)

## Details/Edge Cases

- supported_variants: ['w8a8']
- architecture_family: qwen3_dense
- variant_scope.notes: 该 profile 聚焦已验证 W8A8 工件；Qwen3-32B 其他精度形态需独立验证。
- resource_guidance.evidence_refs: docs/source/tutorials/models/Qwen3-Dense.md, tests/e2e/nightly/single_node/models/configs/Qwen3-32B.yaml, tests/e2e/nightly/single_node/models/configs/Qwen3-32B-Int8.yaml
