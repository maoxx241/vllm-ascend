# Typed KB tables v3

This pass extends the typed intermediate layer with adapter-oriented tables.

## New tables

- `adapter_workitems`
- `adapter_dry_run_fixtures`
- `adapter_dry_run_results`
- `feature_gate_contracts`

## Purpose

The goal of v3 is to move selector/runtime integration from an abstract plan into an executable pre-production contract.

## Adapter workitems

Each workitem groups one or more binding candidates by target. This is the same grouping used in earlier integration planning, but now each workitem is backed by explicit dry-run fixtures.

## Dry-run fixtures

Each fixture contains:

- `binding_id`
- `target`
- `predicate_kind`
- `selector_state`
- `runtime_state`
- `expected_outcome`

The allowed outcomes remain:

- `pass`
- `unsupported`
- `conflict`
- `needs_runtime_probe`
- `needs_clarification`

## Results table

`adapter_dry_run_results` stores the actual adapter outcome for each compiled fixture. A fully matched run means that adapter semantics are stable enough to regression-check before touching production wiring.

## Feature gates

v3 also introduces gate contracts:

- a tools-only dry-run gate
- a production-shadow gate for future selector/runtime wiring
- a variant-breadth caveat policy gate

## Non-goal

v3 still does not modify production selector/runtime code. It only prepares the seam so that future wiring can happen behind a feature gate instead of through ad hoc heuristics.
