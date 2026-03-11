---
name: vllm-ascend-developer-assistant
description: Route development, design-analysis, debugging, adaptation, sync, release, operator, and performance tasks through shared indexes and imported code knowledge.
---

# vLLM Ascend Developer Assistant (E1)

## Purpose

Classify engineering tasks and select the correct shared knowledge path before deeper execution.

This is one of the only two top-level entry skills. Do not expose Composer or Atomic skills as direct user entry points when this skill can perform the first-hop routing.

## Read Order

1. This file only.

## Conditional Reads

Load anything else only after first-hop routing succeeds.

- `../_shared/task-index.md` only if the route is ambiguous
- `../_shared/code-knowledge-map.md` only for code-path follow-up
- `../_shared/knowledge-governance/generated/imported_knowledge_report.json` only for a lightweight confidence or coverage summary
- `../_shared/knowledge-governance/generated/task_skill_index.json` only when the downstream family is ambiguous
- `../_shared/knowledge-governance/generated/design_analysis_index.json` only for concrete design or graph surfaces
- `../_shared/knowledge-governance/generated/imported_knowledge_manifest.json` only for symbol, API, code-location, or repo-ownership questions
- `../vllm-ascend-perf-assistant/SKILL.md` only after a perf request includes enough evidence for real analysis
- `../vllm-ascend-debug-assistant/SKILL.md` only after a debug request includes logs, traces, or a concrete failure phase

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

## First-Hop Routing

1. Use only the user wording and explicit evidence already present in the request to choose the primary task type.
2. For `deployment` or `env_bootstrap`, route to `vllm-ascend-deployment-assistant`.
3. For `performance_analysis`, do not open downstream skills unless the request includes at least three of:
   - baseline/current metrics
   - profiling artifact paths
   - graph/eager comparison
   - topology or parallelism info
   - model and quantization
   - reproduction shape or request mix
4. For `debugging`, do not open downstream skills unless the request includes a log line, stack trace, or concrete failure phase.
5. For `design_analysis` or `model_adaptation`, do not load heavy indexes until the request names a concrete surface, API, model family, or code area.
6. If the request is thin, stop after:
   - `Primary task type`
   - `Route chain`
   - `What is missing`
   - `Smallest next step`
7. Only after the request crosses the evidence threshold should you open downstream Composer or Atomic skills.
8. If the request is about knowledge maintenance, index drift, import refresh, or source/provenance regeneration, route to `vllm-ascend-knowledge-index-maintainer`.

## Deepening Rules

- For code-path questions, use `imported_knowledge_manifest.json` and `code-knowledge-map.md` together.
- If a referenced entry is `validated_with_gap`, keep the answer but surface the gaps explicitly.
- If a referenced entry is not importable, fall back to local code truth and point to `../_shared/knowledge-governance/provenance/verification_manifest.json`.

## Guardrails

- Do not treat `validated_with_gap` as a high-confidence default without mentioning the gaps.
- Prefer exact code evidence over stale docs when they conflict.
- Do not invent task types or skill names outside the shared indexes.
- Do not collapse `design_analysis` into generic development notes; keep it explicit.
- Do not eagerly load shared indexes or downstream skill docs for a short first-hop request.
