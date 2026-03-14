# Selector/runtime shadow wiring v0

## Scope

This pass adds a shadow-only seam under tools/ that can be attached to real
selector/runtime entrypoints behind `VLLM_ASCEND_TYPED_KB_SELECTOR_RUNTIME_ADAPTER`.

## Why shadow-only first

The typed-KB adapter semantics are now explicit and fixture-checked, but runtime
variant coverage is not yet broad enough to justify default-on enforcement.
A shadow envelope lets the branch collect evidence without mutating primary
selector behavior.

## Contract

Input:
- primary selector decision
- selector state
- runtime state
- selector binding candidates
- optional binding filter

Output:
- unchanged primary decision
- shadow envelope with:
  - applicable binding ids
  - per-binding typed-KB result
  - shadow status
  - recommendation
  - counts
  - notes

## Gate behavior

When the gate is disabled:
- no shadow evaluation occurs
- primary decision passes through untouched

When the gate is enabled:
- applicable bindings are evaluated with the adapter shim
- results are summarized into `shadow_ok`, `shadow_blocking`, or
  `shadow_incomplete`
- the primary decision is still preserved

## Current limitation

This is still a tools-layer seam. The target branch must patch its actual
selector/runtime entrypoints to call into the seam behind the feature gate.
