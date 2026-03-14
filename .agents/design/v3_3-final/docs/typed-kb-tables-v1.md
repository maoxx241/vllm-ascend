# Typed KB tables v1

This pass extends the v0 typed intermediate tables with two new capabilities:

1. explicit **version provenance tables**
2. first-pass **compiled constraint predicates**

## Why v1 exists

v0 proved that runtime samples, hardware families, trait families, role topologies,
and high-recall rule families can be compiled into stable intermediate tables. The
remaining gap was that two critical concerns were still implicit:

- cross-sample version drift was only visible in free-form notes
- core constraint families were still represented as family-level guard buckets

v1 addresses those gaps without binding anything into selector/runtime yet.

## New tables

### `runtime_stack_observations`
One row per observed runtime sample. These rows preserve:

- Python / torch / torch_npu / CANN versions
- logical vs physical device counts
- die-per-card observations
- vllm / vllm-ascend import paths
- repo git quality flags
- provenance quality (`repo_git_verified`, `runtime_import_only`, ...)

### `import_surface_observations`
One row per observed runtime import surface. These rows preserve:

- `vllm.envs` and `vllm_ascend.envs` key counts
- CLI discovery vs CLI failure counts
- sampled import-surface class counts

### `import_surface_shapes`
Per-class surface rows for key config classes. These rows make multi-version drift
explicit rather than collapsing it.

### `version_conflict_rows`
Cross-sample divergence rows. These rows do not attempt to reconcile conflicts into
one canonical truth. Instead they attach a reconciliation policy to each dimension.

### `compiled_constraint_predicates`
A first typed predicate layer compiled from high-value branch-guard evidence.
This table targets core families where reliable semantics can be recovered.

## Current scope of compiled predicates

The pass currently compiles typed predicates for these areas:

- `pp` vs `pcp` incompatibility
- decode-role `pp == 1`
- `world_size == tp * pp * pcp`
- `prefill_tp_size % decode_tp_size == 0`
- `prefill_tp_size >= decode_tp_size`
- `enable_kv_nz` requiring MLA and PD decode role
- static-kernel requiring `npugraph_ex`
- `xlite` incompatibility with speculative decoding and PP
- `cp_kv_cache_interleave_size == block_size` under PD+KV-pool scope
- GQA-related divisibility around `num_key_value_heads` and `tp_size`
- `enable_shared_expert_dp` coupling with SP / TP / DP contexts
- `oproj_tensor_parallel_size` graph-mode + PD decode scope

## What v1 still does *not* do

- It does not bind predicates into selector/runtime.
- It does not collapse multi-version class shapes into a single canonical config surface.
- It does not claim runtime-sample completeness beyond the current A2 + A3 observations.

## Merge posture

v1 improves the merge gate, but the gate remains closed until:

1. runtime sample breadth improves
2. compiled predicates are bound through a selector/runtime integration design
