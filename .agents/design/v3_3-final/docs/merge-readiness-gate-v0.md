# Merge readiness gate v0

## Purpose

Provide an explicit gate for when the KB evidence work is ready to be handed to Codex as a larger integration package.

This gate is stricter than the table-compilation pass itself.

## Current checks

1. observed runtime families include both A2 and A3
2. typed intermediate tables compile successfully
3. core rule-family coverage includes the expected high-value constraint families
4. role-topology presence includes prefill/decode/producer/consumer
5. trait-family depth is non-trivial
6. provenance survives into the compiled tables

## Current status

At this point the gate is still **not green** for a full runtime integration handoff.

Reason:

- typed rule compilation is still family-level
- runtime coverage is still narrow
- multi-version reconciliation is not yet compiled

## Interpretation

A `false` gate here does **not** mean the evidence work is invalid.
It means:

- merge of evidence infrastructure is reasonable
- merge of final selector/runtime rule binding is premature

## Next unblockers

1. compile final typed predicates from `constraint_rule_families`
2. reduce noisy role-state families into scoped rule families
3. fold version provenance policy into the intermediate tables
4. widen runtime sample coverage where practical
