# Runtime sample breadth v0

This pass quantifies runtime sample breadth at family and variant granularity.

## Current status

- hardware family coverage: complete for A2 and A3
- hardware variant coverage: incomplete

Current strict variant coverage is intentionally computed against the hardware taxonomy seed. With the present samples:

- expected variants: 7
- strict observed variants: 1
- partial observed variants: 3

## Why variant-level coverage matters

Family-level coverage is enough to establish A2 vs A3 duality, but it is not enough to collapse:

- A2 32G vs 64G memory bins
- A2 20-AICore vs 24-AICore bins
- A3 40-AICore vs 48-AICore bins

These differences should not be promoted into hard runtime rules until more variant-level probes exist.

## Output of this pass

- `runtime_family_coverage_rows`
- `runtime_variant_coverage_rows`
- `runtime_sampling_targets`

The sampling target list exists to keep the breadth blocker explicit instead of leaving it as an informal note.
