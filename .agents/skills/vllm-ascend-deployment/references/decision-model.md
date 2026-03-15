# Deployment decision model

## Scenario

A deployment scenario is always the product of:

- model
- hardware
- input length distribution
- SLA / TPOT
- deployment form

Do not compress it into a single label too early.

## Primary scenario axis

TPOT / SLA is the primary decision axis:

- `TPOT <= 30ms` => low latency
- `TPOT >= 50ms` => high throughput
- `30ms < TPOT < 50ms` => needs alignment

Model depth or structure can be used only as a sanity check, and only as A2/A3 experience. It does not replace SLA.

## Default deployment form

The default is `single_instance`.

This means a single serving instance can still use DP to partition across the local hardware. It does **not** imply multiple independent replicas.

## Feature policy

`MTP`, `FULL_DECODE_ONLY`, and quantized deployment are ordinary deployment controls, not experimental labels by themselves.

Treat them as default-on when there is positive evidence that the chosen model / weight / hardware path supports them.
