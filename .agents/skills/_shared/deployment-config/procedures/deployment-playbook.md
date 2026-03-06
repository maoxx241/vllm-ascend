---
knowledge_id: deployment-config.deployment-playbook
domain: deployment-config
knowledge_type: procedure
summary: Deterministic deployment steps for weak-model-friendly skill outputs.
applicable_vllm_versions: [">=0.15.0", "<0.17.0"]
applicable_cann_versions: [">=8.0.0"]
last_verified: "2026-03-06"
watch_files:
  - "tests/e2e/nightly/single_node/models/configs/Qwen3-32B-Int8.yaml"
  - "tests/e2e/nightly/single_node/models/configs/Qwen3-Next-80B-A3B-Instruct-W8A8.yaml"
  - "examples/run_dp_server.sh"
  - "docs/source/tutorials/models/Qwen3-Dense.md"
  - "docs/source/tutorials/models/Qwen3-Next.md"
depends_on:
  - "../concepts/feature-semantic-dictionary.md"
source_commit: "workspace-head"
freshness: "fresh"
---

# Deployment Playbook

## Goal

Generate a runnable deployment package with stable output structure for weak reasoning models.

## Deterministic Workflow

1. Normalize user text to canonical features.
2. Select model profile (`qwen3-32b-w8a8` by default).
3. Check model-feature compatibility matrix and block unsupported features.
4. Render `start.sh`, `validate.sh`, `rollback.sh`, and `deployment_plan.json`.
5. Validate `/v1/models` then one `/v1/chat/completions` request.
6. If failure, return actionable rollback and one next action.

## Fixed Output Sections

Every skill response must contain exactly these blocks:

1. 参数表
2. 命令块
3. 验证块
4. 风险块

For blocked features, `风险块` must contain:

- blocked feature name
- reason why blocked
- one fallback option

## Quick Commands

Normalization:

```bash
python .agents/skills/vllm-ascend-deployment-assistant/scripts/normalize_terms.py \
  --text "帮我开图并且开w8a8，tp4部署qwen3"
```

Package rendering:

```bash
python .agents/skills/vllm-ascend-deployment-assistant/scripts/render_deploy_package.py \
  --text "帮我开图并且开w8a8，tp4部署qwen3" \
  --model-profile qwen3-32b-w8a8 \
  --output-dir /tmp/vllm_deploy_pkg
```

## Primary and Backup Profiles

- Primary: `qwen3-32b-w8a8`
- Backup: `qwen3-next-80b-a3b-instruct-w8a8`

## Failure Modes and Single-Step Mitigation

- Port conflict: change port and rerun `start.sh`.
- Missing model path: set `--model-path` and rerender.
- Ambiguous features: ask one clarification then rerender.

Back to [INDEX](../../../INDEX.md).
