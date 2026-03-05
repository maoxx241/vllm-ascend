---
knowledge_id: deployment-config.feature-semantic-dictionary
domain: deployment-config
knowledge_type: concept
summary: Natural-language to canonical deployment feature mapping for weak models.
applicable_vllm_versions: [">=0.15.0", "<0.17.0"]
applicable_cann_versions: [">=8.0.0"]
last_verified: "2026-03-06"
watch_files:
  - "docs/source/user_guide/feature_guide/quantization.md"
  - "docs/source/user_guide/feature_guide/graph_mode.md"
  - "docs/source/user_guide/feature_guide/context_parallel.md"
  - "docs/source/tutorials/models/Qwen3-Dense.md"
  - "docs/source/tutorials/models/Qwen3-Next.md"
depends_on:
  - "../../../INDEX.md"
source_commit: "workspace-head"
freshness: "fresh"
---

# Feature Semantic Dictionary (P0+P1)

Use this dictionary as the single source for term normalization.

## Entry Schema

Each feature entry follows:
`canonical_feature, zh_aliases, en_aliases, slang_aliases, cli_flags, prerequisites, incompatibilities, examples`.

## Canonical Entries

### 1) quantization

- canonical_feature: `quantization`
- zh_aliases: `量化`, `开量化`, `int8量化`, `w8a8`
- en_aliases: `quantization`, `int8`, `w8a8`
- slang_aliases: `压模型`, `压权重`
- cli_flags: `--quantization ascend`
- prerequisites: quantized weights available
- incompatibilities: none hard, but can conflict with unsupported kernels
- examples:
  - `给我开w8a8`
  - `enable quantization`

### 2) graph_mode

- canonical_feature: `graph_mode`
- zh_aliases: `图模式`, `开图`, `全图`, `图加速`
- en_aliases: `graph mode`, `cudagraph`, `full decode`
- slang_aliases: `抓图`
- cli_flags: `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`
- prerequisites: model path supports graph capture
- incompatibilities: some dynamic paths may require eager fallback
- examples:
  - `开图跑服务`
  - `enable graph mode`

### 3) int4_quantization

- canonical_feature: `int4_quantization`
- zh_aliases: `int4量化`, `w4a4`, `4bit量化`
- en_aliases: `int4`, `w4a4`, `int4 quantization`, `4bit`
- slang_aliases: `开int4`, `开4bit`
- cli_flags: profile-dependent, no universal safe flag in this demo package
- prerequisites: dedicated int4 model artifact and validated kernel path
- incompatibilities: may be blocked by profile compatibility matrix
- examples:
  - `qwen3-32b-w8a8能开int4吗`
  - `use w4a4`

### 4) tensor_parallel

- canonical_feature: `tensor_parallel`
- zh_aliases: `张量并行`, `tp并行`, `切tp`
- en_aliases: `tensor parallel`, `tp`
- slang_aliases: `横切并行`
- cli_flags: `--tensor-parallel-size <N>`
- prerequisites: NPU count >= TP size
- incompatibilities: invalid topology or missing communication env
- examples:
  - `我想开tp4`
  - `set tp=4`

### 5) data_parallel

- canonical_feature: `data_parallel`
- zh_aliases: `数据并行`, `dp并行`, `切dp`
- en_aliases: `data parallel`, `dp`
- slang_aliases: `副本并行`
- cli_flags: `--data-parallel-size <N>`
- prerequisites: multi-process/multi-node planning
- incompatibilities: wrong dp address / rpc port config
- examples:
  - `开dp`
  - `use data parallel`

### 6) expert_parallel

- canonical_feature: `expert_parallel`
- zh_aliases: `专家并行`, `ep并行`
- en_aliases: `expert parallel`, `ep`
- slang_aliases: `moe并行`
- cli_flags: `--enable-expert-parallel`
- prerequisites: MoE model
- incompatibilities: non-MoE models (mark as not-applicable)
- examples:
  - `moe模型开ep`
  - `enable EP`

### 7) prefill_decode_disaggregation

- canonical_feature: `prefill_decode_disaggregation`
- zh_aliases: `预填充解码分离`, `pd分离`, `prefill-decode分离`
- en_aliases: `prefill decode disaggregation`, `pd disaggregation`
- slang_aliases: `P节点D节点`
- cli_flags: `--data-parallel-*` and connector configs (profile-specific)
- prerequisites: multi-node/network setup
- incompatibilities: single-host quick demo without connector
- examples:
  - `我要做PD分离`
  - `enable prefill decode disaggregation`

### 8) prefix_cache

- canonical_feature: `prefix_cache`
- zh_aliases: `前缀缓存`, `开缓存`
- en_aliases: `prefix cache`, `automatic prefix caching`
- slang_aliases: `复用前缀`
- cli_flags: default on; disable by `--no-enable-prefix-caching`
- prerequisites: repeated-prefix workload
- incompatibilities: none hard
- examples:
  - `保留prefix cache`
  - `turn on prefix caching`

### 9) context_parallel

- canonical_feature: `context_parallel`
- zh_aliases: `上下文并行`, `长上下文并行`, `cp并行`
- en_aliases: `context parallel`, `cp`
- slang_aliases: `长序列并行`
- cli_flags: deployment-profile specific additional config
- prerequisites: long context use-case and compatible topology
- incompatibilities: invalid model/backend combo
- examples:
  - `长上下文场景开cp`
  - `enable context parallel`

### 10) lora

- canonical_feature: `lora`
- zh_aliases: `lora`, `lora适配`, `挂lora`
- en_aliases: `lora adapter`
- slang_aliases: `外挂LoRA`
- cli_flags: `--enable-lora`
- prerequisites: LoRA artifacts available
- incompatibilities: missing adapter files
- examples:
  - `把lora挂上`
  - `enable lora`

### 11) speculative_decode

- canonical_feature: `speculative_decode`
- zh_aliases: `投机解码`, `草稿解码`, `spec decode`
- en_aliases: `speculative decoding`, `mtp`
- slang_aliases: `猜词加速`
- cli_flags: `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`
- prerequisites: model/backend support
- incompatibilities: unsupported model path
- examples:
  - `开投机`
  - `enable speculative decoding`

### 12) sleep_mode

- canonical_feature: `sleep_mode`
- zh_aliases: `休眠模式`, `空闲休眠`
- en_aliases: `sleep mode`
- slang_aliases: `省电模式`
- cli_flags: `--enable-sleep-mode`
- prerequisites: compatible runtime
- incompatibilities: none hard
- examples:
  - `空闲时休眠`
  - `enable sleep mode`

### 13) weight_prefetch

- canonical_feature: `weight_prefetch`
- zh_aliases: `权重预取`, `预取权重`
- en_aliases: `weight prefetch`
- slang_aliases: `提前拉权重`
- cli_flags: `--additional-config '{"weight_prefetch_config":{"enabled":true}}'`
- prerequisites: supported model/profile
- incompatibilities: may increase memory pressure
- examples:
  - `开权重预取`
  - `enable weight prefetch`

## Short Trigger Rules (Weak Model)

- If phrase contains `并行` but not explicit `tp/dp/ep/cp`, mark as ambiguous.
- If phrase contains `开图`, map to `graph_mode`.
- If phrase contains `w8a8/int8`, map to `quantization`.
- If phrase contains `int4/w4a4/4bit`, map to `int4_quantization`.
- If phrase contains unknown feature words, return up to 3 candidate features and ask one clarification.

## Ambiguity Fallback

Only ask one question:
- `你说的并行更偏向 TP、DP 还是 EP？`
- `你说的开图是 FULL_DECODE_ONLY 还是先走 eager 对比？`

Back to [INDEX](../../../INDEX.md).
