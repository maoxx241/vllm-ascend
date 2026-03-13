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

## Addendum F: Critical-Slot Clarification and Defaulting

- The frozen v3.3 package defines a first-class `confirmation gate`, but it does
  not define a first-class clarification gate for missing critical slots.
- This gap must be handled as an implementation-time hardening rule, not by
  silently guessing route-changing inputs.
- `what_is_missing` and `open_questions` are not sufficient by themselves unless
  the runtime also blocks artifact synthesis when a critical slot is unresolved.

### F.1 Principle

- If a missing or conflicting slot can change:
  - family selection
  - execution mode
  - artifact path
  - topology strategy
  - performance baseline
  then the system must not silently guess.
- If a missing slot does not change the route and a local default is safe, the
  runtime may continue, but it must surface the assumption explicitly.

### F.2 Slots That Must Trigger Clarification or Route Backoff

- Hardware choice when different hardware families would select materially
  different documented baselines or optimization profiles.
- Artifact-path choice, including `native fp8` vs `convert_then_deploy`.
- Topology requests that are self-conflicting, such as physical-card count that
  cannot satisfy the requested `tp/dp/ep`.
- Cases where the requested topology is present but the parallelism axis is
  under-specified and different choices would change the deployment artifact or
  expected-performance answer.
- Any request where the route is still unstable enough that the family should be
  `design_analysis` instead of `deployment_execution` or `performance_analysis`.

### F.3 Slots That May Use a Default

- Runtime hardware may be used as the default target only when:
  - the user did not explicitly name hardware
  - the route does not branch across multiple materially different artifact
    paths
  - the answer explicitly states that it is assuming the current runtime
    environment
- Defaulting must never rewrite:
  - `native fp8` into `w8a8`
  - a locked topology into a different topology
  - an unresolved route into a documented deployment command

### F.4 Output Rules Until a First-Class Clarification Gate Exists

- If a critical slot is unresolved, the runtime must prefer:
  - `design_analysis + spec_plan_workflow`, or
  - an explicit `unknown` / conditional answer
- In that state, deployment and performance capabilities must not emit a
  “final-looking” script or single-point expectation.
- If a local default is used, the output must say so directly, for example:
  - “hardware not specified; assuming current runtime A2”
  - “topology not specified; returning documented best-performance baseline”

### F.5 Implementation Guidance

- Treat clarification as a global concern, not a model-specific patch.
- Enforce it in normalization, routing, and capability gates instead of skill
  prose.
- Skills may remind the caller to use runtime-selected assumptions, but they
  must not independently decide which missing slots are safe to guess.
