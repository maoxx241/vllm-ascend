---
name: vllm-ascend-test-matrix-planner
description: Internal atomic skill for building minimal profiling or performance experiment matrices on vLLM Ascend. Invoke only after routing through vllm-ascend-developer-assistant.
---

# vLLM Ascend Test Matrix Planner (A16)

## Purpose

Convert a fuzzy performance hypothesis into a small, defensible experiment matrix.

## Entry Policy

This is not a top-level entry skill. It must be invoked through `vllm-ascend-developer-assistant`.

## Read Order

1. `../_shared/deployment-config/references/global-parameter-feature-map.md`
2. `../_shared/knowledge-governance/generated/design_analysis_index.json`
3. `references/matrix-heuristics.md`

## Workflow

1. Freeze the baseline:
   - model and checkpoint
   - topology
   - sequence lengths / request mix
   - graph/eager mode
   - profiling on/off state
2. Pick exactly one control dimension for the next experiment batch.
3. Design the smallest matrix that can falsify the current hypothesis:
   - usually 3 to 8 runs
   - one baseline row
   - one changed variable per row
4. Define success criteria before proposing the runs:
   - throughput or latency threshold
   - utilization target
   - queue/execute ratio shift
   - memory headroom change
5. State what artifacts must be saved from each run.

## Output Contract

- `Baseline`
- `Hypothesis under test`
- `Control variable`
- `Experiment matrix`
- `Required artifacts`
- `Decision rule`

## Guardrails

- Do not vary multiple major dimensions in the same row unless the goal is explicit interaction testing.
- Do not propose wide sweeps when one binary isolation step would answer the question.
- Keep the matrix small enough that the team will actually run it.
