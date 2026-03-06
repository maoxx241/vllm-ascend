---
knowledge_id: ascend-foundation.env-bootstrap-baseline
domain: ascend-foundation
knowledge_type: procedure
summary: Baseline environment bootstrap and health checks for vllm-ascend deployment.
applicable_vllm_versions: [">=0.15.0", "<0.17.0"]
applicable_cann_versions: [">=8.0.0"]
last_verified: "2026-03-06"
watch_files:
  - "AGENTS.md"
  - "vllm-ascend/AGENTS.md"
  - "vllm-ascend/requirements.txt"
  - "vllm/pyproject.toml"
depends_on:
  - "../../../INDEX.md"
source_commit: "workspace-head"
freshness: "fresh"
---

# Env Bootstrap Baseline

## Four Fixed Sections

`vllm-ascend-env-bootstrap` must always output these four sections:

1. 环境检查
2. 安装动作
3. 健康检查
4. 修复建议

## Baseline Checks

- shell env loaded from `.bashrc`
- python/pip path consistency
- NPU env variables available
- vllm and vllm-ascend editable install status

## Minimal Command Set

```bash
# vLLM
export VLLM_TARGET_DEVICE=empty
pip install -e . --no-build-isolation

# vLLM Ascend
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

- Do not modify business code during bootstrap.
- If any check fails, return exact failed command and one repair command.

Back to [INDEX](../../../INDEX.md).
