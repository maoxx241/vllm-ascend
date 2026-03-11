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
3. `references/lazy-routing.md`

## Conditional Reads

Load these only after first-hop task classification:

- `../_shared/code-knowledge-map.md` for code-path questions or when the downstream skill needs likely code surfaces
- `../_shared/knowledge-governance/generated/imported_knowledge_report.json` for lightweight evidence-status or coverage summary
- `../_shared/knowledge-governance/generated/task_skill_index.json` only when the downstream chain is ambiguous
- `../_shared/knowledge-governance/generated/design_analysis_index.json` only for concrete design/perf/graph/model-adaptation surfaces
- `../_shared/knowledge-governance/generated/imported_knowledge_manifest.json` only for symbol, API, code-location, or repo-ownership questions

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

1. Use `task-index.md` and `references/lazy-routing.md` to pick the primary task chain before loading any heavyweight generated index.
2. If the user request is thin, ambiguous, or artifact-free, stop at first-hop routing:
   - return the primary task type
   - return the route chain
   - ask for the minimum evidence bundle
   - do not load heavyweight generated indexes yet
3. Load `imported_knowledge_report.json` only when a lightweight evidence-status summary adds value.
4. Load `task_skill_index.json` only when the downstream family is ambiguous.
5. For design-heavy requests with a concrete surface, consult `design_analysis_index.json`.
6. For code-path questions, use `imported_knowledge_manifest.json` and `code-knowledge-map.md` together.
7. If a referenced entry is `validated_with_gap`, keep the answer but surface the gaps explicitly.
8. If a referenced entry is not importable, fall back to local code truth and point to `../_shared/knowledge-governance/provenance/verification_manifest.json`.
9. If the request is about knowledge maintenance, index drift, import refresh, or source/provenance regeneration, route to `vllm-ascend-knowledge-index-maintainer`.

## Guardrails

- Do not treat `validated_with_gap` as a high-confidence default without mentioning the gaps.
- Prefer exact code evidence over stale docs when they conflict.
- Do not invent task types or skill names outside the shared indexes.
- Do not collapse `design_analysis` into generic development notes; keep it explicit.
- Do not eagerly load heavyweight generated indexes for a short first-hop request.
