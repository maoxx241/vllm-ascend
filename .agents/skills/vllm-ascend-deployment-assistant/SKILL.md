---
name: vllm-ascend-deployment-assistant
description: Deploy models on vLLM-Ascend from natural language requests using deterministic term normalization, fixed output templates, and runnable start/validate packages.
---

# vLLM Ascend Deployment Assistant (E2)

## Purpose

Convert natural-language deployment requests into deterministic deployment artifacts.

This is one of the only two top-level entry skills. Environment bootstrap, compatibility checks, and deployment subflows should be reached through this assistant rather than exposed directly as first-hop entry points.

## Read Order

1. `../_shared/INDEX.md`
2. `../_shared/ai-foundation/INDEX.md`
3. `../_shared/deployment-config/concepts/feature-semantic-dictionary.md`
4. `../_shared/ai-foundation/indexes/topic-index.json`
5. `../_shared/ai-foundation/indexes/term-alias-index.json`
6. `../_shared/ai-foundation/indexes/view-index.json`
7. `../_shared/ai-foundation/indexes/rule-index.json`
8. `../_shared/deployment-config/references/global-parameter-feature-map.md`
9. `../_shared/deployment-config/references/global-parameter-verification-report.md`
10. `../_shared/deployment-config/references/global-parameter-combination-guide.md`
11. `../_shared/deployment-config/procedures/deployment-playbook.md`
12. `../_shared/vllm-ascend-core/concepts/model-feature-compatibility-matrix.md`
13. `../_shared/troubleshooting/procedures/unsupported-feature-cases.md`
14. `references/output-schema.md`

## Weak-Reasoning Mode (Mandatory)

- Never skip steps.
- Make only one decision at a time.
- If ambiguous, return up to 3 candidates and ask one clarification.
- All commands must be copy-paste runnable.

## Two-Step Execution

### Step 1: Intent Normalization

Run:

```bash
python .agents/skills/vllm-ascend-deployment-assistant/scripts/normalize_terms.py \
  --text "<user_input>"
```

Expected output interface:

```json
{
  "intent": "deploy_model",
  "features": ["graph_mode", "quantization"],
  "confidence": 0.91,
  "missing_slots": [],
  "clarification_question": ""
}
```

If `missing_slots` is not empty, ask exactly one clarification question then re-run normalization.

### Step 2: Deploy Package Rendering

Run:

```bash
python .agents/skills/vllm-ascend-deployment-assistant/scripts/render_deploy_package.py \
  --text "<user_input>" \
  --model-profile qwen3-32b-w8a8 \
  --output-dir /tmp/vllm_deploy_pkg
```

Supported model profiles:

- `qwen3-32b-w8a8` (default)
- `qwen3-next-80b-a3b-instruct-w8a8` (backup)

## Compatibility Rules (Mandatory)

- Always report blocked features with reasons.
- Do not silently downgrade unsupported features.
- For `qwen3-32b-w8a8`:
  - `int4_quantization` must be blocked.
  - `expert_parallel` must be blocked.

## Required Final Output Format

Always use these six sections in this order:

1. 参数表
2. 命令块
3. 验证块
4. 风险块
5. 证据块
6. 冲突告警块

Template details are in `references/output-schema.md`.

## Guardrails

- Do not edit business code while deploying.
- Do not execute destructive git operations.
- If a feature is not applicable, mark it clearly with reason.
- For low-confidence/upstream-delta evidence, keep recommendation but emit explicit warning.
