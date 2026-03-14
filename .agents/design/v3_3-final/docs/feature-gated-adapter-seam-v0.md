# Feature gated adapter seam v0

This pass defines the gate contract for future selector/runtime integration.

## Gate set

### 1. `typed_kb_adapter_dry_run`

- env: `VLLM_ASCEND_TYPED_KB_ADAPTER_DRY_RUN`
- default: `false`
- scope: tools only

Purpose: execute dry-run fixtures without affecting production behavior.

### 2. `typed_kb_selector_runtime_adapter`

- env: `VLLM_ASCEND_TYPED_KB_SELECTOR_RUNTIME_ADAPTER`
- default: `false`
- scope: production shadow

Purpose: allow selector/runtime requests to be shadow-evaluated by the typed-KB adapter before any default behavior changes are introduced.

### 3. `typed_kb_variant_caveat_policy`

- env: `VLLM_ASCEND_TYPED_KB_VARIANT_CAVEAT_POLICY`
- default: `true`
- scope: policy / warning

Purpose: keep hardware-variant caveats explicit until strict runtime sample coverage grows beyond the current `1/7` state.

## Why a shadow gate comes before production enablement

Runtime breadth is still incomplete at the exact hardware-variant level. That makes a shadow-only gate the correct next step:

- it allows production paths to observe typed-KB decisions
- it avoids silently changing runtime decisions while variant breadth is still narrow
- it creates a clean promotion path from shadow evaluation to authoritative evaluation later
