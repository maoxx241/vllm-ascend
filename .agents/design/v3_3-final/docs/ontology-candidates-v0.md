# Ontology candidates v0

This document summarizes the first compiler pass from the unified evidence catalog into ontology candidates.

## Candidate inventory

The current compiler emits:

- runtime samples: `2`
- hardware families: `2`
- hardware variants: `7`
- model trait families: `11`
- role topology groups: `28`
- high-value branch guard families: `17`

These counts come from `artifacts/kb_inventory/ontology_candidates_v0.json`.

## Why this compiler pass is useful

The previous passes produced evidence but not candidate ontology objects. This pass keeps provenance explicit while grouping evidence into ontology-ready buckets.

The current grouped objects already support the following next-step compiler targets:

- `runtime_sample`
- `hardware_family`
- `hardware_variant`
- `model_trait`
- `role_topology`
- `branch_guard_rule_family`

## Current runtime candidates

The two observed runtime samples already separate the hardware families by direct probe evidence:

- `A2_910B4_32G`
  - logical devices = `8`
  - physical cards = `8`
  - die/card estimate = `1`
- `A3_mainline_corrected`
  - logical devices = `16`
  - physical cards = `8`
  - die/card estimate = `2`

This is enough to justify treating runtime samples as a first-class ontology object instead of flattening them into a single `hw=A2/A3` label too early.

## Current model-trait candidates

The compiler currently groups trait evidence into 11 model-trait families, including:

- `architecture`
- `mla`
- `moe`
- `mtp`
- `gqa`
- `gqa_partitioning`
- `long_context`

This is the minimum useful trait layer required for later strategy/topology reasoning.

## Current role-topology candidates

The compiler currently groups role-topology evidence into 28 distinct groups.

The most important point is structural: role topology is already evidence-backed and cannot be represented faithfully by a single flat topology tuple.

Examples already visible in the grouped candidates include:

- `prefill` and `decode` tuples from documentation tables
- `producer` and `consumer` CLI/YAML tuples
- `prefill` and `decode` nested configs from `kv_connector_extra_config`
- explicit `prefill_tp_size` / `decode_tp_size` code references

## Current branch-guard candidates

The compiler groups high-value branch guards into 17 families, including:

- `parallel.pp`
- `parallel.tp`
- `parallel.dp`
- `parallel.pcp`
- `parallel.dcp`
- `parallel.cp_interleave`
- `feature.kv_nz`
- `feature.shared_expert_dp`
- `graph.ascend_compilation`
- `connector.kv.extra`
- `trait.num_key_value_heads`
- `trait.num_experts`

This is the right shape for the next compiler pass: rules should be compiled from grouped evidence families, not from raw grep hits.

## Remaining limitation

The branch-guard families are still high-recall and slightly noisy. In particular, generic `prefill` / `decode` mentions are common. The next pass should rank or filter those families before promoting them into hard constraints.

## Recommended next pass

1. promote `runtime_sample`, `hardware_family`, and `hardware_variant` into typed KB tables
2. promote `model_trait` into typed KB tables
3. normalize `role_topology` groups into typed topology tuples with provenance refs
4. compile selected `branch_guard_rule_family` groups into deduplicated constraint objects
