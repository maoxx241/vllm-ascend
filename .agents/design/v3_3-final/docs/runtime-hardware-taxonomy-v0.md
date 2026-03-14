# Runtime hardware taxonomy v0

This note records the first hardware-family split driven by runtime probes plus operator-supplied Ascend topology facts.

## Why this note exists

The previous evidence pipeline established the software/config control surfaces, but it still treated hardware too coarsely. The submitted `910B4 32G` sample plus the operator notes on `910B*` and `910C` make it clear that the KB needs more than `hw=A2|A3`.

## Minimum fields to carry forward

The next compiler stage should be able to represent, separately:

- `soc_family`
- `soc_variant`
- `device_name_from_probe`
- `memory_gb_per_card`
- `die_per_card`
- `aicore_per_die`
- `aicore_per_card`
- `inter_node_hccs`
- `observed_device_count`
- `provenance`

## Required provenance split

Hardware facts must not all share the same trust level.

- `probe_derived`: from `runtime_tuple.json`
- `repo_derived`: from code/docs/tests
- `operator_supplied`: user/operator notes not yet independently corroborated
- `operator_supplied_unverified`: planning hints not yet safe as hard constraints

## Initial family sketch

### A2

Operator-supplied seed:

- `910B4 32G`
- `910B4 64G`
- `910B3 64G`
- `910B2 64G`
- `910B1 64G`
- single card, single die
- no inter-node HCCS

### A3

Operator-supplied seed:

- `910C 64G`
- single card, two die
- inter-node HCCS available
- two AICore bins observed in the family (20/die and 24/die)

## Probe implications

The runtime probe now needs more than `npu-smi info`. It must also capture:

- `npu-smi info -m`
- `npu-smi info -l`
- `npu-smi info -t topo`
- `npu-smi info -t hccs`
- `npu-smi info -t hccs-bw`
- `npu-smi info -t board`
- `npu-smi info -t product`
- `npu-smi info -t memory`
- `npu-smi info -t work-mode`
- `npu-smi info -t phyid-remap`

These commands are required to move from a flat model-name observation to a structured hardware-topology model.
