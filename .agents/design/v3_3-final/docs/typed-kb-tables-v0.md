# Typed KB tables v0

## Goal

Compile the current provenance-aware evidence catalog into stable intermediate tables before binding any predicate schema into selector/runtime logic.

This pass is intentionally positioned between:

1. `evidence_catalog_v3.jsonl`
2. final typed KB / runtime constraint compiler

## Inputs

Primary input:

- `artifacts/kb_inventory/evidence_catalog_v3.jsonl`

Evidence sources already represented in the catalog include:

- runtime probes (A2 + A3 samples)
- operator-supplied hardware family seeds
- AST branch guards
- HF/HF-text trait surfaces
- role-scoped topology extraction

## Output tables

### 1. `hardware_runtime_samples`

One row per observed runtime sample. Key fields:

- `sample_id`
- `hardware_family`
- `hardware_variant`
- `logical_device_count`
- `physical_card_count`
- `die_per_card_estimate`
- `memory_gb_per_logical_device`
- `logical_hbm_aggregate_gb`
- `topology_observability`
- `torch/torch_npu/CANN` versions
- provenance fields

### 2. `hardware_families`

One row per canonical hardware family seed. Key fields:

- `hardware_family`
- `aliases`
- `die_per_card`
- `inter_node_fabric`
- `inter_node_hccs`
- `moe_prefill_ep_preference`
- `variants`
- `observed_sample_ids`

### 3. `hardware_observed_probe_seeds`

Probe-adjacent rows that should remain separate from canonical family rows until more runtime corroboration exists.

### 4. `model_trait_families`

Trait-family aggregations over HF/HF-text surfaces. Key fields:

- `trait_family`
- `trait_category`
- `top_fields`
- `source_kind_counts`
- `provenance_counts`
- `example_refs`

### 5. `role_topology_patterns`

Role-scoped topology patterns normalized over axis aliases such as:

- `tp_size -> tp`
- `dp_size -> dp`
- `prefill_tp_size -> prefill_tp`
- `decode_tp_size -> decode_tp`

### 6. `constraint_rule_families`

Family-level guard compilation over AST `if/assert/raise` evidence. Key fields:

- `rule_family`
- `rule_category`
- `normalized_condition`
- `normalized_message`
- `root_ref_counts`
- `role_counts`
- `guard_kind_counts`
- `provenance_counts`
- `example_refs`

## Why this pass exists

Without this pass, the system jumps from raw evidence directly into ontology candidates and then risks binding unstable or overly noisy rules into runtime logic.

Typed tables provide:

- stable row shapes
- preserved provenance
- normalized axes
- explicit merge gates
- a cleaner handoff into the final predicate/rule compiler

## Current limitations

- `constraint_rule_families` is still high-recall and family-level.
- Some role-state families (`role.prefill`, `role.decode`, etc.) remain noisy and should not yet be treated as final KB predicates.
- Runtime sample coverage is still narrow.
- Multi-version reconciliation is not yet compiled into the tables.

## Merge policy

This pass is allowed to merge as evidence infrastructure, but **not** as final selector/runtime truth.

Anything downstream that consumes `constraint_rule_families` must treat it as intermediate evidence until the typed predicate compiler lands.
