# Control-surface evidence v3

## Purpose

This pass extends the v2 control-surface work in three directions:

1. Normalize runtime probe samples across probe schema versions.
2. Extract AST-aware branch guards from code instead of relying only on lexical anchors.
3. Extract role-scoped topology and HF trait surfaces as first-class evidence objects.

The objective is still **evidence-first**, not final ontology freeze.

## New artifacts

- `artifacts/kb_inventory/runtime_sample_summaries.jsonl`
- `artifacts/kb_inventory/runtime_sample_normalization_summary.md`
- `artifacts/kb_inventory/branch_guard_rules_ast.jsonl`
- `artifacts/kb_inventory/branch_guard_summary.md`
- `artifacts/kb_inventory/hf_trait_surfaces.jsonl`
- `artifacts/kb_inventory/hf_trait_summary.md`
- `artifacts/kb_inventory/role_topologies.jsonl`
- `artifacts/kb_inventory/role_topology_summary.md`
- `artifacts/kb_inventory/evidence_catalog_v3.jsonl`
- `artifacts/kb_inventory/evidence_catalog_v3_summary.md`
- `artifacts/kb_inventory/hardware_taxonomy_seed_v1.yaml`

## Quantitative summary

The v3 pass yields the following evidence counts:

- runtime sample summaries: `2`
- AST branch guards: `2172`
- HF trait surfaces: `611`
- role-topology records: `157`
- unified evidence catalog: `3255`

The unified catalog count is consistent with the component counts plus the hardware seed rows:

\[
2 + 2172 + 611 + 157 + 3 = 2945
\]

That formula would be incorrect because AST branch guards are expanded by family in the unified catalog.
The actual catalog count comes from the normalized records emitted by `build_evidence_catalog.py`, not a simple raw-row sum. The raw branch-guard rows are `2172`, but the catalog contains `2482` `branch_guard` evidence rows after per-family normalization.

The catalog breakdown is:

- `runtime_probe`: `2`
- `repo_static_ast`: `2482`
- `repo_static_trait`: `611`
- `repo_static_role_topology`: `157`
- `operator_supplied`: `2`
- `probe_derived_plus_operator_supplied`: `1`

## Runtime sample normalization

Two runtime samples are now normalized to one canonical summary schema:

- `A2_910B4_32G`
- `A3_mainline_corrected`

Observed results:

- A2 sample:
  - logical devices = `8`
  - physical cards = `8`
  - die/card estimate = `1`
  - logical HBM total per device = `32768 MB`
  - logical aggregate HBM = `262144 MB`
- A3 sample:
  - logical devices = `16`
  - physical cards = `8`
  - die/card estimate = `2`
  - logical HBM total per device = `65536 MB`
  - logical aggregate HBM = `1048576 MB`
  - `hccs` query support = `true`

Derived ratios are consistent with the family distinction:

\[
\text{A3 logical/card ratio} = 16 / 8 = 2
\]

\[
\text{A2 logical/card ratio} = 8 / 8 = 1
\]

These are not final family rules yet, but they are strong evidence inputs for hardware canonicalization.

## AST-aware branch tracing

The new AST pass records `if`, `assert`, and `raise` sites that mention high-value control-surface and trait tokens.

Top family counts include:

- `role.decode`: `665`
- `role.prefill`: `473`
- `cache.block_size`: `263`
- `feature.specdecode`: `204`
- `parallel.pcp`: `166`
- `parallel.dcp`: `122`
- `trait.hf_config`: `94`
- `trait.num_experts`: `93`

This matters because the repository already contains non-trivial guard structure beyond simple public knobs. The AST pass gives us a machine-readable evidence layer for those guards.

## HF trait surfaces

The trait extraction pass records both code and documentation evidence for `hf_config` / `hf_text_config` consumers and trait-bearing model fields.

Top trait families include:

- `moe`: `252`
- `architecture`: `82`
- `mla`: `75`
- `mtp`: `71`
- `gqa`: `14`
- `gqa_partitioning`: `6`

High-value trait fields already visible in the current mainline evidence include:

- `num_key_value_heads`
- `num_experts`
- `architectures`
- `model_type`
- `rope_scaling`
- `max_position_embeddings`

This is sufficient to justify promoting `model_trait` to a first-class ontology object in the next pass.

## Role-scoped topology extraction

The role-topology pass currently emits `157` records across four evidence kinds:

- `doc_role_tuple`: `50`
- `kv_connector_extra_config`: `56`
- `server_cmd_flags`: `42`
- `role_axis_reference`: `49`

The current extraction already recovers structured role evidence such as:

- `prefill` vs `decode` tuple layouts from docs
- `producer` vs `consumer` deployment commands from nightly YAML
- `prefill` / `decode` nested topology from `kv_connector_extra_config`
- `prefill_tp_size` / `decode_tp_size` references from code

This is enough to justify a role-scoped topology object instead of flattening everything into a single `(tp, dp, ep)` tuple.

## Evidence model implications

The v3 pass supports the following ontology backlog items:

- `runtime_sample`
- `hardware_family`
- `hardware_variant`
- `model_trait`
- `branch_guard_rule`
- `role_topology`
- `validated_deployment_tuple`
- `evidence_provenance`

The key architectural conclusion is unchanged:

**Ontology should be compiled from typed evidence objects, not inferred directly from raw docs or raw probe outputs.**

## Immediate next step

The next pass should focus on compiling the typed evidence into ontology candidates, not on widening raw control-surface enumeration again.

Concretely:

1. Promote `runtime_sample` and `hardware_family/variant` into typed ontology candidates.
2. Promote `model_trait` and `role_topology` into typed ontology candidates.
3. Compile `branch_guard_rule` from AST evidence into a normalized rule object.
4. Keep provenance explicit in every compiled fact.
