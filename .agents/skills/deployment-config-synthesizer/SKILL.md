# deployment-config-synthesizer

Atomic deployment skill that turns a validated deployment capsule into a
minimal config and runtime-env draft.

Use the capsule as the source of truth. If the topology requested by the user
is not documented, synthesize a conservative config that still preserves the
requested topology, and mark it as inferred / unvalidated. If topology is not
specified, use the documented best-performance baseline.
