# Ontology backlog v0

This document captures the minimum viable ontology objects justified by the current evidence base.

## Design principle

Every ontology object below must be compilable from evidence with explicit provenance.
The compiler may rank or merge evidence, but it must not erase where each fact came from.

## Candidate object types

### 1. `evidence_provenance`

Purpose:
- track whether a fact came from runtime probe, runtime import surface, repo code/docs/tests, or operator note

Required fields:
- `provenance_type`
- `source_path`
- `source_line`
- `sample_label` (if runtime-derived)
- `confidence`
- `status` (`observed`, `derived`, `operator_supplied`, `pending_corroboration`)

### 2. `runtime_sample`

Purpose:
- represent one observed runtime tuple without forcing it into a hardware family too early

Required fields:
- `sample_label`
- `torch_version`
- `torch_npu_version`
- `cann_version`
- `logical_device_count`
- `physical_card_count`
- `chip_count`
- `die_per_card_estimate`
- `device_names`
- `hbm_total_mb_values`
- `query_support` (`hccs`, `phyid_remap`, `product`, `work_mode`)

### 3. `hardware_family`

Purpose:
- canonical family abstraction such as `A2` / `A3`

Required fields:
- `family_name`
- `aliases`
- `die_per_card`
- `inter_node_hccs`
- `canonicalization_rules`
- `provenance`

### 4. `hardware_variant`

Purpose:
- distinguish memory/AICore bins within a family

Required fields:
- `family_name`
- `variant_name`
- `hbm_gb`
- `aicore_per_card` or `aicore_per_die`
- `notes`
- `provenance`

### 5. `model_trait`

Purpose:
- represent model-side structural properties that interact with routing and topology

Initial examples:
- `mla`
- `gqa_partitioning`
- `moe`
- `mtp`
- `long_context`
- `architecture`

Required fields:
- `trait_name`
- `trait_family`
- `trigger_fields` (e.g. `num_key_value_heads`, `num_experts`, `architectures`)
- `evidence_refs`

### 6. `branch_guard_rule`

Purpose:
- represent code-side conditions and explicit constraints

Required fields:
- `rule_id`
- `condition_expr`
- `message_expr`
- `root_refs`
- `roles`
- `rule_kind` (`if`, `assert`, `raise`)
- `provenance`

### 7. `role_topology`

Purpose:
- capture role-scoped topology instead of flattening into one tuple

Required fields:
- `role`
- `axes`
- `multiplier`
- `evidence_kind`
- `connector_context`
- `provenance`

### 8. `validated_deployment_tuple`

Purpose:
- represent a deployment tuple that is seen in docs, tests, or nightly config

Required fields:
- `model_or_workload`
- `role_topologies`
- `connector`
- `quantization`
- `graph_mode`
- `runtime_family_hint`
- `validation_source`
- `provenance`

## What should not be frozen yet

The following should remain evidence-level rather than ontology-level in the immediate next pass:

- CLI surface completeness
- long-tail env vars with no high-value consumers
- low-adoption experimental features (`xlite`, `cpp`)
- operator-supplied performance preferences that still lack repo/runtime corroboration

## Compiler order

The recommended compiler order is:

1. `runtime_sample`
2. `hardware_family` / `hardware_variant`
3. `model_trait`
4. `branch_guard_rule`
5. `role_topology`
6. `validated_deployment_tuple`

This keeps provenance intact and avoids over-freezing the ontology before the evidence has converged.
