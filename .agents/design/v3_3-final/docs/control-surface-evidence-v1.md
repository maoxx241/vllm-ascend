# Control-surface evidence v1

## Objective

Shift the knowledge extraction flow from shallow feature lookup to a control-surface-first evidence pipeline. The first pass inventories the roots that actually drive behavior changes in `vllm` and `vllm-ascend`, then links those roots back to code, docs, examples, and tests.

## Scope of this pass

This pass inventories:

- CLI arguments from `vllm/vllm/engine/arg_utils.py`
- Environment variables from `vllm/vllm/envs.py` and `vllm-ascend/vllm_ascend/envs.py`
- Config fields from the main config object graph
- `additional_config` top-level and nested dict-tunnel keys from the public docs
- First-pass document command seeds extracted from markdown launch examples

This pass does **not** yet attempt a complete semantic consumer graph. Coverage remains presence-based.

## Lifecycle buckets

- `core_baseline`
- `core_differentiator`
- `emerging`
- `legacy_experimental`
- `unclassified`

Graph-family handling in this pass:

- `graph.acl` stays in `core_baseline`
- `graph.npugraph_ex` moves to `emerging`
- `graph.xlite` moves to `legacy_experimental`

## First-pass inventory summary

Generated from `artifacts/kb_inventory/root_inventory_summary.json`:

- Total roots: `723`
- CLI roots: `172`
- Env roots: `254`
- Config-field roots: `257`
- Dict-tunnel roots: `40`
- Doc command seeds: `40`
- Unique doc-derived normalized roots: `84`

## Drift observations

Current automated drift findings:

- Docs-only top-level additional-config key: `additional_config.enable_npugraph_ex`
- Code-only top-level additional-config key: `additional_config.ascend_fusion_config`

This confirms that `npugraph_ex` alias normalization must be explicit in the next phase.

## Known limitations of v1

1. Consumer tracing is lexical and token-based.
2. Coverage scores are presence-based, not proof of semantic completeness.
3. `unclassified` remains large because many generic vLLM env/config surfaces have not been bucketed yet.
4. Nested dict-tunnel keys are inventoried from docs, but code-side nested coverage is still partial.

## Immediate next step

Use the current evidence base to do a second pass over the highest-value roots only:

- `async_scheduling`
- `cudagraph_mode`
- `pipeline/tensor/data/pcp/dcp`
- `kv_transfer` / `ec_transfer`
- `enable_kv_nz`
- `enable_shared_expert_dp`
- `layer_sharding`
- `finegrained_tp_config`
- `eplb_config`
- `ascend_compilation_config.*`

That second pass should replace lexical tracing with scoped semantic tracing for role-sensitive branches and constraint expressions.
