# deployment_execution

Family-local skill for `feature-policy-resolver` and deployment-oriented atomic
work packages.

Required order:
- Start from `deployment-intake`, not raw repo search.
- Resolve baseline/policy from the capsule first.
- Then run `deployment-config-synthesizer`.
- Finish with `deployment-artifact-packager`.

Do not fabricate a single-card script when the capsule says the path is
undocumented or unsupported. Return the documented TP4 / 4-NPU baseline and
state that the single-card path is outside the documented launch matrix.
