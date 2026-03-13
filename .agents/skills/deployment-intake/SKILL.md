# deployment-intake

Canonical deployment intake wrapper. It turns a raw request into
`selector_seed/v3` plus an intake-stage or follow-on deployment plan when a
query is required.

Use `runtime.py` before opening docs. If the request asks for a script or
config, continue from intake to `feature-policy-resolver`,
`deployment-config-synthesizer`, and `deployment-artifact-packager`.

Treat card counts, topology locks, parallelism hints, and priority hints as
normalized runtime selectors. Do not restate hardware facts here; rely on the
capsule and shared runtime instead.
