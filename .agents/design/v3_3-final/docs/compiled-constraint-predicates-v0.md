# Compiled constraint predicates v0

This pass introduces a typed predicate layer on top of high-value branch-guard evidence.

## Goal

Convert a narrow, high-value subset of rule families from:

- free-form branch guards
- family-level grouped constraints

into:

- typed predicate rows with normalized payloads
- stable targets that later selector/runtime design can bind against

## Why only a subset

The AST guard pass is intentionally high-recall. Many guard families are noisy,
role-overloaded, or too implementation-specific to bind directly. v0 therefore only
compiles predicates where semantics are clear and provenance is strong.

## Predicate kinds used in v0

- `incompatible_axes`
- `axis_equals`
- `formula`
- `divisible_by`
- `gte`
- `requires_trait`
- `requires_role`
- `requires_feature`
- `requires_mode`
- `incompatible_features`
- `incompatible_feature_axis`
- `equals`
- `multiple_of`
- `coupled_axis_feature`

## Examples

- `parallel.pp_vs_pcp` → incompatible when both axes are greater than 1
- `role.decode.pp` → decode role requires `pp == 1`
- `feature.enable_kv_nz` → requires MLA + PD decode role
- `graph.enable_static_kernel` → requires `graph.npugraph_ex`
- `parallel.cp_kv_cache_interleave_size` → equals `block_size` under PD+KV-pool context

## Non-goals

- no selector/runtime binding in this pass
- no attempt to compile every guard family
- no precedence/conflict resolution between predicates yet

## Expected follow-up

The next integration layer should:

1. attach predicate scope semantics to selector/runtime objects
2. define precedence / conflict handling
3. combine predicates with multiversion provenance partitioning
