# Selector/runtime adapter dry-run v0

This pass introduces executable acceptance fixtures for the selector/runtime adapter seam.

## Coverage

- prioritized workitems: `14`
- fixture cases: `41`
- outcome classes exercised: `5`

## Why this matters

Earlier passes could describe adapter bindings, but they could not prove that the intended outcomes were deterministic. The dry-run layer closes that gap.

## Coverage pattern

The fixture set intentionally exercises:

- success cases
- dependency failures
- numeric failures
- selector conflicts
- runtime probe gaps
- clarification gaps

## Example derived binding

`parallel.world_size` is now evaluated as:

\[
world\_size = tp 	imes pp 	imes pcp
\]

In the positive fixture:

\[
tp = 4,\; pp = 2,\; pcp = 2
\]

So:

\[
world\_size = 4 	imes 2 	imes 2 = 16
\]

When `runtime.logical_device_count = 16`, the derived binding passes.
When `runtime.logical_device_count = 8`, the derived binding is marked unsupported because the derived world size exceeds observed runtime capacity.

## Implication

The remaining integration risk is no longer formula ambiguity. It is the last-mile production wire-up and runtime breadth policy.
