---
name: vllm-ascend-knowledge-index-maintainer
description: Validate, regenerate, and evolve the shared vLLM/vLLM-Ascend knowledge source, provenance, and generated retrieval indexes. Use when knowledge entries change, imports are added, domain ownership drifts, or `_shared` indexes need to be rebuilt.
---

# Knowledge Index Maintainer (A18)

## Purpose

Own the machine-verifiable lifecycle of shared knowledge under `_shared`.

## Read Order

1. `../_shared/INDEX.md`
2. `../_shared/task-index.md`
3. `../_shared/code-knowledge-map.md`
4. `../_shared/knowledge-governance/provenance/execution_state.json`
5. `../_shared/knowledge-governance/provenance/verification_handoff.md`
6. `../_shared/knowledge-governance/contracts/knowledge_domain_registry.json`
6. `references/validation_rules.json`

## Workflow

1. Treat `../_shared/vllm-upstream/references/source/knowledge/`, `../_shared/vllm-ascend-core/references/source/knowledge/`, and `../_shared/integration-core/references/source/knowledge/` as the canonical domain-owned source roots.
2. Treat `../_shared/knowledge-governance/` as the only governance root for contracts, provenance, and unified generated indexes.
3. Run `scripts/validate_shared_knowledge.py` to adjudicate `domain_scope`, refresh `knowledge_domain` and `source_hash`, rebuild provenance, and regenerate unified retrieval indexes.
3. If interrupted or the context folds, resume from `execution_state.json`, not from conversation history.
4. Use `scripts/test_validate_shared_knowledge_outputs.py` and `scripts/test_skill_scenario_coverage.py` after regeneration.
5. For knowledge maintenance requests, report both source/provenance health and generated index health.

## Guardrails

- Do not reintroduce a root-level `knowledge-base/` dependency.
- Keep unified generated consumer paths stable under `../_shared/knowledge-governance/generated/`.
- Only mark `domain_scope=both` when the fact truly depends on both upstream vLLM and vLLM-Ascend.
- Prefer preserving verified source and provenance over deleting evidence.
