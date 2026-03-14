# Runtime hardware taxonomy v1

This update incorporates the corrected `npu-smi` probe behavior and the stronger A3 sample captured with that probe.

## Why v1 exists

The v0 note identified the need for richer hardware fields, but it still assumed that many `npu-smi info -t ...` forms could be queried as a single global command.

That assumption was wrong in practice.

The corrected probe now uses a two-stage procedure:

1. discover card/chip scope from `npu-smi info` and `npu-smi info -l`
2. issue card-scoped or chip-scoped queries as required

This is a meaningful design correction, not just a parser tweak.

## Corrected probe semantics

### Base queries

The following can be collected once:

- `npu-smi info`
- `npu-smi info -m`
- `npu-smi info -l`
- `npu-smi info -t topo`
- `npu-smi info -t product`
- `npu-smi info -t work-mode`

### Scoped queries

These must be issued with explicit scope when the runtime requires it:

- per-chip:
  - `npu-smi info -t hccs -i <card> -c <chip>`
  - `npu-smi info -t hccs-bw -i <card> -c <chip> -time <ms>`
- per-card:
  - `npu-smi info -t board -i <card>`
  - `npu-smi info -t memory -i <card>`
  - `npu-smi info -t phyid-remap -i <card>`

Unsupported commands should be recorded as `unsupported_by_runtime`, not interpreted as probe failure.

## Stronger A3 observations from the corrected probe

The new A3 sample contributes these probe-derived facts:

- physical cards: 8
- logical devices / chips: 16
- chips per card: 2
- local die pairs connected by `SIO`
- broader off-diagonal chip relationships reported as `HCCS_SW`
- HCCS scoped queries succeed per chip
- `product`, `work-mode`, and `phyid-remap` are unsupported on this stack

## What this changes in the KB plan

The KB should no longer model hardware as only `A2` vs `A3`. It should be able to carry, separately:

- `soc_family`
- `soc_variant`
- `device_name_from_probe`
- `card_count`
- `logical_device_count`
- `chips_per_card`
- `topology.local_pair_relation`
- `topology.cross_pair_relation`
- `hccs_present`
- `memory_report_scope`
- `memory_gb_per_card` (only when semantically supported)
- `aicore_per_die`
- `aicore_per_card`
- `provenance`

## New caution on memory semantics

The corrected A3 sample reports `65536 MB` HBM per logical device/chip. Because the device is 2-die-per-card, the KB must not automatically reinterpret that value as a physical-card total.

Until the board/product semantics are pinned down across versions, the safer field is:

- `hbm_total_mb_per_logical_device`

not:

- `memory_gb_per_card`

## Practical consequence

The runtime probe is now good enough to seed hardware taxonomy facts directly, but those facts must retain scope and provenance.
