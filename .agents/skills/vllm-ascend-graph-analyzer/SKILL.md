---
name: vllm-ascend-graph-analyzer
description: Internal atomic skill for graph-mode, capture/replay, compile, and shape-stability analysis on vLLM Ascend. Invoke only after routing through vllm-ascend-developer-assistant.
---

# vLLM Ascend Graph Analyzer (A7)

## Purpose

Isolate graph-related bottlenecks or failure modes from general performance noise.

## Entry Policy

This is not a top-level entry skill. It must be invoked through `vllm-ascend-developer-assistant`.

## Read Order

1. `../_shared/code-knowledge-map.md`
2. `../_shared/knowledge-governance/generated/design_analysis_index.json`
3. `../_shared/knowledge-governance/generated/imported_knowledge_manifest.json`
4. `references/graph-signals.md`

## Workflow

1. Confirm that graph mode is relevant:
   - graph enabled
   - capture/replay logs exist
   - eager fallback changes the symptom
2. Classify the graph symptom:
   - capture overhead
   - replay mismatch
   - shape drift / batch invariance break
   - compile instability
   - graph gives lower steady-state benefit than expected
3. Map the symptom to shared knowledge entries and code surfaces.
4. Separate:
   - warmup/capture cost
   - replay cost
   - shape or scheduling instability
5. Return the most likely graph-specific suspect and what eager/shape-isolation rerun would validate it.

## Output Contract

- `Graph symptom`
- `Relevant evidence`
- `Most likely graph-specific suspect`
- `Likely code surfaces`
- `Isolation rerun`

## Guardrails

- Do not blame graph mode if the same symptom exists in eager mode.
- Do not use generic operator explanations when the evidence points to shape instability or replay mismatch.
- Keep graph diagnosis separate from pure scheduling or communication issues unless the evidence crosses both surfaces.
