# Codex branch integration plan v0

## Objective

Integrate the shadow-only typed-KB selector/runtime seam into the target branch
without changing default behavior.

## Required branch changes

1. Apply the overlay bundle.
2. Keep `VLLM_ASCEND_TYPED_KB_SELECTOR_RUNTIME_ADAPTER` default-off.
3. Identify the real selector/runtime request evaluation path in the branch.
4. After the primary selector decision is computed, call the shadow seam.
5. Attach the returned shadow envelope to logs/diagnostics only.
6. Do not use shadow results to mutate the primary decision yet.

## Required validation

- Run `run_selector_runtime_adapter_dry_run.py`
- Run `run_selector_runtime_shadow_cases.py`
- Confirm gate-disabled passthrough remains intact
- Confirm gate-enabled shadow results match the checked-in expectations

## Caveat policy

Variant breadth remains incomplete. The integration should carry forward the
existing caveat policy and avoid any default-on behavior until more runtime
samples are collected.
