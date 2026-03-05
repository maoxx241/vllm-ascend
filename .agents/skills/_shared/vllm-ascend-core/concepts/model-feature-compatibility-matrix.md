---
knowledge_id: vllm-ascend-core.model-feature-compatibility-matrix
domain: vllm-ascend-core
knowledge_type: concept
summary: Model-profile level compatibility matrix used by deployment assistant.
applicable_vllm_versions: [">=0.15.0", "<0.17.0"]
applicable_cann_versions: [">=8.0.0"]
last_verified: "2026-03-06"
watch_files:
  - "docs/source/tutorials/models/Qwen3-Dense.md"
  - "docs/source/tutorials/models/Qwen3-Next.md"
  - "docs/source/user_guide/support_matrix/supported_models.md"
  - "tests/e2e/nightly/single_node/models/configs/Qwen3-32B-Int8.yaml"
  - "tests/e2e/nightly/single_node/models/configs/Qwen3-Next-80B-A3B-Instruct-W8A8.yaml"
depends_on:
  - "../../../INDEX.md"
  - "../../../deployment-config/concepts/feature-semantic-dictionary.md"
source_commit: "workspace-head"
freshness: "fresh"
---

# Model Feature Compatibility Matrix

This matrix drives hard blocking behavior in deployment package generation.

## Profiles

- `qwen3-32b-w8a8`: dense, W8A8 quantized profile.
- `qwen3-next-80b-a3b-instruct-w8a8`: MoE-like Next profile for backup demo.

## Model Knowledge Fields (for deterministic inference)

| Profile | `has_moe_layers` | `num_experts` | `fixed_weight_format` | `supported_variants` | `feature_min_npu_count` |
| --- | --- | --- | --- | --- | --- |
| `qwen3-32b-w8a8` | `false` | `0` | `w8a8` | `["w8a8"]` | `data_parallel=8, context_parallel=8` |
| `qwen3-next-80b-a3b-instruct-w8a8` | `true` | `80` | `w8a8` | `["w8a8"]` | `data_parallel=8, context_parallel=8` |

Inference rules:

1. If `has_moe_layers=false`, then `expert_parallel` is `hard_block`.
2. If `int4`/`w4a4` is not in `supported_variants`, then `int4_quantization` is `hard_block`.
3. If requested feature has `feature_min_npu_count` and current `npu_count` is lower, then mark as `downgraded` and skip applying that feature.

## Feature Compatibility

| Canonical feature | qwen3-32b-w8a8 | qwen3-next-80b-a3b-instruct-w8a8 | Rule |
| --- | --- | --- | --- |
| `quantization` | Supported (`--quantization ascend`) | Supported (`--quantization ascend`) | Default quantized flow |
| `int4_quantization` | **Blocked** | **Blocked** | No validated int4 path in demo profiles |
| `graph_mode` | Supported | Supported | `FULL_DECODE_ONLY` |
| `tensor_parallel` | Supported | Supported | TP defaults to 4 |
| `data_parallel` | Conditional | Conditional | Requires enough cards/topology |
| `expert_parallel` | **Blocked** | Supported | Dense model cannot use EP |
| `prefill_decode_disaggregation` | Conditional | Conditional | Multi-node connector dependent |
| `prefix_cache` | Supported | Supported | Enabled by default in this package |
| `context_parallel` | Conditional | Conditional | Needs topology and long-context target |
| `lora` | Supported | Supported | Adapter files required |
| `speculative_decode` | Conditional | Conditional | Backend/model support required |
| `sleep_mode` | Supported | Supported | Runtime support required |
| `weight_prefetch` | Supported | Supported | Can increase memory pressure |

## Explicit Error Cases

- Request: `qwen3-32b-w8a8 + int4`
  - Result: blocked, reason: profile fixed to W8A8.
- Request: `qwen3-32b-w8a8 + ep`
  - Result: blocked, reason: no MoE layers (`has_moe_layers=false`), EP not applicable.

Back to [INDEX](../../../INDEX.md).
