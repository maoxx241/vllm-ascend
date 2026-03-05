---
knowledge_id: troubleshooting.unsupported-feature-cases
domain: troubleshooting
knowledge_type: procedure
summary: Known unsupported or not-applicable feature combinations and fallback actions.
applicable_vllm_versions: [">=0.15.0", "<0.17.0"]
applicable_cann_versions: [">=8.0.0"]
last_verified: "2026-03-06"
watch_files:
  - "docs/source/user_guide/support_matrix/supported_features.md"
  - "docs/source/user_guide/support_matrix/supported_models.md"
  - "docs/source/tutorials/models/Qwen3-Dense.md"
depends_on:
  - "../../INDEX.md"
  - "../../vllm-ascend-core/concepts/model-feature-compatibility-matrix.md"
source_commit: "workspace-head"
freshness: "fresh"
---

# Unsupported Feature Cases

## Hard-block examples

### Case 1: `qwen3-32b-w8a8 + int4`

- Why blocked: model profile is already W8A8 quantized; int4 path is not validated for this profile.
- Fallback: keep W8A8, or switch to a dedicated W4A4 model artifact.

### Case 2: `qwen3-32b-w8a8 + EP`

- Why blocked: Qwen3-32B-W8A8 profile is dense, EP is MoE-specific.
- Fallback: remove EP, keep TP/DP tuning.

## Soft-block examples

### Case 3: low-card setup + data parallel/context parallel

- Why soft-block: topology/resources insufficient for expected scaling.
- Fallback: remain TP-focused in single-node demo.

### Case 4: speculative decode on unsupported path

- Why soft-block: backend/model support may be incomplete.
- Fallback: disable speculative decode and run base generation path.

Back to [INDEX](../../INDEX.md).
