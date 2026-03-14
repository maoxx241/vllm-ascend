# Selector runtime binding v0

This document defines the intermediate binding layer between typed KB predicates and future selector/runtime integration.

## Layering

- evidence catalog
- typed KB tables v0/v1
- compiled selector binding candidates (this pass)
- selector/runtime integration (future pass)

## Binding primitives

Each binding candidate carries:

- `binding_kind`
- `target`
- `selector_slots`
- `runtime_slots`
- `derived_slots`
- `payload`
- `provenance_type`
- `integration_state`

## Binding kinds currently compiled

- `selector_conflict`
- `numeric_constraint`
- `dependency_constraint`
- `derived_value`

## Examples

### Conflict binding

`parallel.pp_vs_pcp`

- selector slots: `parallel.pp`, `parallel.pcp`
- runtime slots: `runtime.parallel_axes`
- meaning: both axes may be observable in requests, but the realized runtime topology must reject the simultaneous active state

### Dependency binding

`feature.enable_kv_nz`

- selector slots: `feature.enable_kv_nz`, `model.trait.mla`
- selector slots: `feature.enable_kv_nz`, `role.decode.present`, `topology.pd_disaggregate`
- runtime slots: `runtime.feature_modes`, `runtime.model_traits`, `runtime.role_topology`

### Derived binding

`parallel.world_size`

- selector slots: `parallel.tp`, `parallel.pp`, `parallel.pcp`
- runtime slots: `runtime.logical_device_count`, `runtime.physical_card_count`
- derived slot: `derived.parallel.world_size`

## Non-goal of this pass

This pass does not mutate runtime code or selector contracts. It only makes the binding layer explicit and serializable.
