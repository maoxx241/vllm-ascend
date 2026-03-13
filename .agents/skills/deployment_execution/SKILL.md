# deployment_execution

Family-local skill for `feature-policy-resolver` and deployment-oriented atomic
work packages.

Required order:
- Start from `deployment-intake`, not raw repo search.
- Resolve baseline/policy from the capsule first.
- Then run `deployment-config-synthesizer`.
- Finish with `deployment-artifact-packager`.
- Use runtime output as the only truth source for both the selected artifact
  path and the selected topology strategy.

Do not silently substitute the user's requested topology with another script.
If the request matches a documented baseline, return that documented script.
If the request does not match a documented baseline, analyze the gap and return
an inferred script that still satisfies the user's requested topology, such as
`single-card`, but mark it as unvalidated and list the risks. If the user did
not specify topology or priority, default to the documented best-performance baseline.

Use the capsule as the source of truth for topology facts and selected
strategies. Do not emit a script unless runtime output contains both a selected
artifact path and a selected strategy. If the selected strategy is
`unknown_or_reroute`, or the selected artifact is `unsupported_requires_choice`,
do not fabricate a script inside this family; reroute to `design_analysis`
instead.
