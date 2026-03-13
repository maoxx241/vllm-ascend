# deployment-intake

Canonical deployment intake wrapper. It turns a raw request into
`selector_seed/v3` plus an intake-stage or follow-on deployment plan when a
query is required.

Use `runtime.py` before opening docs. If the request asks for a script or
config, continue from intake to `feature-policy-resolver`,
`deployment-config-synthesizer`, and `deployment-artifact-packager`.

On A3, parse physical card-count requests first. `1 card = 2 logical NPUs`,
so `4 cards` is an `8 logical NPU` request, not `TP4` by default.
