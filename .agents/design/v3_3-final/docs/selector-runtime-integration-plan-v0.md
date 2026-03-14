# Selector/runtime integration plan v0

## Current state

- typed KB rows through v2 exist
- selector binding candidates are compiled
- runtime family coverage is complete at family level
- runtime variant coverage remains incomplete
- selector/runtime integration is still an intermediate layer, not an implemented adapter

## Goal of this pass

Turn `selector_binding_candidates` into an implementable adapter contract without changing existing runtime schemas yet.

## Scope

1. Normalize binding candidates into integration work items.
2. Define adapter responsibilities by slot kind.
3. Define dry-run acceptance checks that can be executed before runtime code changes.
4. Keep runtime variant breadth as a separate blocker.

## Work items

### W1. Slot-to-runtime adapter contract

Map each selector slot family to a runtime source:

- `parallel.*` -> `runtime.parallel_axes`
- `role.*` -> `runtime.role_topology`
- `feature.*` -> `runtime.feature_modes`
- `model.trait.*` -> `runtime.model_traits`
- `hardware.*` -> `runtime.hardware_profile`
- `cache.*` -> `runtime.cache_profile`
- `derived.*` -> adapter-computed values

### W2. Binding evaluation phases

Bindings should be evaluated in this order:

1. `selector_conflict`
2. `dependency_constraint`
3. `numeric_constraint`
4. `derived_value`

Rationale: conflict and dependency failures should short-circuit before any derived value is materialized as authoritative.

### W3. Dry-run acceptance checks

A dry run must verify for each binding candidate:

- all selector slots have a declared type
- all runtime slots have a declared source surface
- derived slots have an explicit formula or derivation hint
- failure mode is deterministic: `conflict`, `unsupported`, `needs_runtime_probe`, or `needs_clarification`

### W4. Runtime integration boundary

This pass does **not** modify selector/runtime production code. It only defines the adapter seam and generates fixtures to validate it.

## Remaining blockers after this pass

- runtime sample breadth for exact hardware variants
- production integration of adapter contract into selector/runtime code path
