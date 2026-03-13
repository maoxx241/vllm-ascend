# Implementation Addenda / Non-Authoritative Supplement

This document records implementation-time constraints and repair rules found
after the v3.3 design package was frozen. It is a local implementation aid,
not a new source of truth.

## Conflict Handling

When this file conflicts with the frozen design package, use the existing
priority order:

1. `schema/*.json`
2. `docs/06-interface-contracts.md`
3. `README.md`
4. `docs/00-09/*.md`
5. `examples/*.json`

This file may clarify implementation constraints, but it must not create new
public contract semantics or override frozen schema.

## Implementation Red Lines

1. `artifact:*` and `strategy:*` are `atom_id` naming conventions only.
   They must not introduce a new `atom_kind` outside the existing schema enum.
2. `ModelSlim` inside `deployment_execution` means runbook generation only.
   The family may synthesize conversion and serving steps, but it must not
   execute quantization tools.

## Addendum A: Physical Card != Logical NPU / Comm Domain

- `physical_cards` is a user-facing topology constraint.
- `logical_npus` is a runtime-facing execution fact.
- On A3, one physical card maps to two logical NPUs, but that fact alone does
  not force a single parallelism strategy.
- Communication and runtime environment selection must be derived from:
  `hw + logical_npus + tp/dp/ep + backend`, not from physical-card count alone.
- Stable skills must not restate topology math as prompt truth; that knowledge
  belongs in KB facts and capsule output.

## Addendum B: Native FP8 != Supported Quantized Artifact

- `native fp8 weight` is not equivalent to `w8a8` or other supported quantized
  artifact paths on A2/A3.
- If the request is for native FP8 weights on A2/A3 and no route has been
  chosen, the system must first explain that native FP8 direct deployment is
  unsupported and then route to `design_analysis`.
- A supported quantized artifact path, such as ModelSlim-generated weights,
  must be modeled as a separate artifact-path decision.
- Runtime normalization must not silently rewrite `fp8` to `w8a8`.

## Addendum C: Deployment Runbook != Tool Execution

- `deployment_execution` may generate:
  - documented serve scripts
  - documented conversion + serve runbooks
  - inferred but unvalidated runbooks
- `deployment_execution` may not run quantization or conversion tools.
- If the user chooses an fp8-origin adaptation path that implies loader,
  backend, or code changes, the request must leave `deployment_execution` and
  enter `adaptation` with confirmation.

## Addendum D: Runtime Truth Source != Skill Text Truth Source

- Runtime objects and capsule output remain the truth source.
- Skill text exists to enforce call order and guardrails for agents that do not
  natively execute the runtime.
- Stable skill text must not become the place where hardware, quantization, or
  deployment truths are maintained.
- If runtime output does not include the selected artifact path or selected
  topology strategy, the skill must not fabricate a script.

## Addendum E: Cross-Agent Hardening

- Cross-agent portability is a repo-local hardening requirement layered on top
  of P6, not a new v3.3 public contract.
- Canonical skill docs must enforce:
  - `runtime.py` first
  - no raw-doc search first
  - no script synthesis without selected runtime objects
  - reroute to `design_analysis` when the route is unresolved
