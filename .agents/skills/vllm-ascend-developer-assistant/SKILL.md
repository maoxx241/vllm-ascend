---
name: vllm-ascend-developer-assistant
description: Route development, design-analysis, debugging, adaptation, sync, release, operator, and performance tasks through shared indexes and imported code knowledge.
---

# vLLM Ascend Developer Assistant (E1)

## Purpose

Classify engineering tasks and select the correct shared knowledge path before deeper execution.

This is one of the only two top-level entry skills. Do not expose Composer or Atomic skills as direct user entry points when this skill can perform the first-hop routing.

## Read Order

1. `../_shared/INDEX.md`
2. `../_shared/task-index.md`
3. `../_shared/code-knowledge-map.md`
4. `../_shared/knowledge-governance/generated/imported_knowledge_report.json`
5. `../_shared/knowledge-governance/generated/task_skill_index.json`
6. `../_shared/knowledge-governance/generated/design_analysis_index.json`
7. `../_shared/knowledge-governance/generated/imported_knowledge_manifest.json`

## Task Classification

Always classify the request into exactly one primary task type first:

- `design_analysis`
- `model_adaptation`
- `debugging`
- `upstream_sync`
- `release_analysis`
- `op_development`
- `performance_analysis`
- `env_bootstrap`
- `deployment`

If the request spans multiple areas, pick the highest-risk primary task and list secondary task types separately.

## Workflow

1. Use `task-index.md` to pick the primary task chain.
2. Load `task_skill_index.json` and `imported_knowledge_report.json` to find the relevant skill family and evidence status.
3. For design-heavy requests, always consult `design_analysis_index.json` before proposing conclusions.
4. For code-path questions, use `imported_knowledge_manifest.json` and `code-knowledge-map.md` together.
5. If a referenced entry is `validated_with_gap`, keep the answer but surface the gaps explicitly.
6. If a referenced entry is not importable, fall back to local code truth and point to `../_shared/knowledge-governance/provenance/verification_manifest.json`.
7. If the request is about knowledge maintenance, index drift, import refresh, or source/provenance regeneration, route to `vllm-ascend-knowledge-index-maintainer`.

## Guardrails

- Do not treat `validated_with_gap` as a high-confidence default without mentioning the gaps.
- Prefer exact code evidence over stale docs when they conflict.
- Do not invent task types or skill names outside the shared indexes.
- Do not collapse `design_analysis` into generic development notes; keep it explicit.
