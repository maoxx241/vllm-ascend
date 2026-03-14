# Typed KB tables v2

This pass extends the intermediate KB with two additional layers:

1. selector/runtime binding candidates
2. runtime sample breadth coverage

## Why v2 exists

Pass7 proved that typed predicates and version provenance can be compiled into stable intermediate tables. The remaining blockers were:

- runtime sample breadth
- selector/runtime binding

v2 does not change selector/runtime code. Instead it compiles the missing intermediate interface so the last remaining selector-related blocker becomes explicitly "runtime integration" rather than "binding still missing".

## New tables

- `selector_slot_catalog`
- `selector_binding_candidates`
- `runtime_binding_requirements`
- `runtime_family_coverage_rows`
- `runtime_variant_coverage_rows`
- `runtime_sampling_targets`

## What is now explicit

The KB now contains an auditable mapping from:

- typed predicates
- role topology patterns
- model trait families
- hardware/runtime samples

into selector-facing slots and runtime-facing requirement rows.

That means targets such as:

- `parallel.pp_vs_pcp`
- `role.decode.pp`
- `parallel.world_size`
- `feature.enable_kv_nz`
- `feature.enable_shared_expert_dp`
- `graph.enable_static_kernel`
- `graph.xlite_vs_parallel.pp`
- `graph.xlite_vs_specdecode`

are no longer trapped in prose notes or family-level summaries.

## Remaining gap

The remaining selector blocker is no longer knowledge extraction. It is final schema/runtime integration:

- emitting these bindings into selector request/response slots
- wiring runtime-side evaluators to consume them
- defining integration precedence when version provenance rows disagree
