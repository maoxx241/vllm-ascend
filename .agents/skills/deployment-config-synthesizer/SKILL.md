# deployment-config-synthesizer

Atomic deployment skill that turns a validated deployment capsule into a
minimal config and runtime-env draft.

Use the capsule as the source of truth. If the topology requested by the user
is not documented, synthesize the documented baseline config and mark the
requested topology as unsupported.
