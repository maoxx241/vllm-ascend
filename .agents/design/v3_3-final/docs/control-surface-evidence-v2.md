# Control-surface evidence v2

This document records the second-pass evidence extraction over the root control surfaces of the combined `vllm + vllm-ascend` workspace.

## What changed in v2

v1 established a denominator. v2 keeps the inventory approach but adds a scoped semantic trace over the highest-value root families. The focus is not on claiming completeness; it is on proving that the important roots can be traced into:
- implementation consumers
- explicit constraints
- docs/tutorials
- tests/examples

## Current outputs

- root inventory: `artifacts/kb_inventory/root_inventory.jsonl`
- document command seeds: `artifacts/kb_inventory/doc_command_seeds.jsonl`
- high-value consumer edges: `artifacts/kb_inventory/consumer_edges.high_value.jsonl`
- derived rules: `artifacts/kb_inventory/derived_rules.high_value.jsonl`
- validation links: `artifacts/kb_inventory/validation_links.high_value.jsonl`
- coverage report: `artifacts/kb_inventory/coverage_report_v2.md`

## Current headline numbers

- inventory roots: 659
- document command seeds: 227
- high-value edges: 6165
- high-value validation links: 2361
- derived rules: 16

## What v2 proved

1. The main serving/runtime path really is concentrated around:
   - `async_scheduling`
   - `compilation_config.cudagraph_mode`
   - `pipeline_parallel_size`
   - `prefill_context_parallel_size`
   - `decode_context_parallel_size`
   - `cp_kv_cache_interleave_size`
   - `kv_transfer_config`
   - `enable_kv_nz`
   - `enable_shared_expert_dp`
   - `finegrained_tp_config`
   - `eplb_config`

2. `graph` has to remain split into lifecycle families:
   - `graph.acl` as core baseline
   - `graph.npugraph_ex` as emerging
   - `graph.xlite` as legacy/experimental

3. Several high-value rules are no longer “implied knowledge”; they are now explicit evidence items. Examples:
   - `pp` and `pcp` cannot both be enabled in the Mooncake KV connector
   - decode-side `pp_size` must be `1`
   - DCP has separate MLA and GQA constraints
   - `cp_kv_cache_interleave_size` must equal `block_size` in KV-transfer scenarios
   - `enable_kv_nz` is MLA-only and PD-decode-only
   - `enable_shared_expert_dp` is gated by additional config + EP + TP + SP
   - static kernel requires `npugraph_ex`
   - Xlite is incompatible with speculative decoding and PP, and requires `block_size=128`

## Confirmed drift

The most important confirmed drift is still:

- docs expose a top-level `additional_config.enable_npugraph_ex`
- the implemented path is `additional_config.ascend_compilation_config.enable_npugraph_ex`

There is also a code-surface item missing from the top-level additional-config table:

- `ascend_fusion_config`

## Remaining boundary

This v2 pass is still root- and anchor-driven. It is enough to shape ontology, but not enough to finalize it. The next pass should add:
- model-trait extraction from `hf_config` / `hf_text_config`
- AST-aware branch tracing for value propagation
- role-scoped topology extraction for PD/KV transfer pipelines
