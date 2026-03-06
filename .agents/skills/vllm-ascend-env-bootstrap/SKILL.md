---
name: vllm-ascend-env-bootstrap
description: Bootstrap and validate vLLM-Ascend runtime environment with deterministic four-section outputs for weak-reasoning models.
---

# vLLM Ascend Env Bootstrap (A1)

## Purpose

Provide a deterministic environment setup path before deployment.

## Read Order

1. `../_shared/INDEX.md`
2. `../_shared/ascend-foundation/procedures/env-bootstrap-baseline.md`

## Mandatory Output Structure

Always return exactly four sections:

1. 环境检查
2. 安装动作
3. 健康检查
4. 修复建议

## Weak-Reasoning Mode

- No skipped steps.
- One decision per step.
- If any command fails, return failed command + one fix command.

## Baseline Command Order

```bash
# In vllm/
export VLLM_TARGET_DEVICE=empty
pip install -e . --no-build-isolation

# In vllm-ascend/
pip install -r requirements.txt
pip install -v -e . --no-build-isolation
```

## Health Checks

```bash
python -c "import vllm; print(vllm.__version__)"
python -c "import vllm_ascend; print('vllm_ascend import ok')"
vllm --help >/dev/null
```

## Guardrails

- Do not modify business code.
- Do not assume model-specific feature flags during bootstrap.
