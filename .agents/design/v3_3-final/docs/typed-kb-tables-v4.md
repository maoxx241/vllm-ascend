# Typed KB tables v4

This pass promotes the typed-KB selector/runtime adapter from dry-run fixtures into a
shadow-only seam contract.

## New tables

- `shadow_selector_requests`
- `shadow_runtime_contexts`
- `shadow_status_policy`
- `shadow_eval_envelopes_enabled`
- `shadow_eval_envelopes_disabled`

## Purpose

The v4 tables are not a production cutover mechanism. They exist to make the
production-shadow feature gate executable and regression-checkable before any
real selector/runtime entrypoint patch is merged.

## Key property

The primary selector decision remains unchanged in both gate states.
The typed-KB adapter only attaches a shadow envelope.

## Status policy

- `pass -> shadow_ok`
- `unsupported/conflict -> shadow_blocking`
- `needs_runtime_probe/needs_clarification -> shadow_incomplete`

## Current merge posture

The remaining blocker is runtime sample breadth. Family-level coverage is in
place, but strict variant coverage remains narrow.
