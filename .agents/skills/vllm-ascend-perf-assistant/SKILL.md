---
name: vllm-ascend-perf-assistant
description: Internal composer skill for vLLM Ascend profiling and performance analysis across offline profiling artifacts, benchmark logs, graph signals, and experiment planning. Invoke only after routing through vllm-ascend-developer-assistant.
---

# vLLM Ascend Perf Assistant (C6)

## Purpose

Turn profiling artifacts and benchmark evidence into a ranked bottleneck analysis, then organize the next tuning or validation loop.

## Entry Policy

This is not a top-level entry skill. It must be invoked through `vllm-ascend-developer-assistant`.

## Read Order

1. `references/perf-taxonomy.md`
2. `references/minimal-evidence-gate.md`
3. `../_shared/task-index.md`

## Conditional Reads

Load these only after the evidence threshold in `references/minimal-evidence-gate.md` is met:

- `../_shared/code-knowledge-map.md` for likely code surfaces
- `../_shared/knowledge-governance/generated/imported_knowledge_report.json` for lightweight evidence-status summary
- `../_shared/knowledge-governance/generated/design_analysis_index.json` for graph, scheduler, parallel, or architecture-heavy perf issues
- `../_shared/knowledge-governance/generated/imported_knowledge_manifest.json` for symbol, API, or code-location follow-up

Avoid `task_skill_index.json` in normal perf analysis unless the chain itself is ambiguous.

## When To Use

Use this skill when the primary task type is `performance_analysis`, especially for:

- throughput or latency regressions
- MFU or utilization analysis
- graph-mode overhead or replay inefficiency
- memory or KV-cache pressure that manifests as performance degradation
- scheduling or batching tradeoff analysis
- profiling output interpretation
- planning the next tuning experiment set

## Workflow

1. Use `references/perf-taxonomy.md` and `references/minimal-evidence-gate.md` to decide whether this is a low-evidence routing response or a real bottleneck analysis.
2. If the request is low-evidence, stop early:
   - classify the perf question
   - list the missing evidence
   - return the smallest next experiment or collection step
   - do not load heavyweight shared knowledge
3. Normalize the evidence bundle:
   - benchmark summary or SLA target
   - profiling artifact type and location
   - model, quantization, topology, and critical flags or env vars
   - baseline vs regression point
   - whether the issue is steady-state, warmup, or request-shape dependent
4. Route the analysis:
   - always start with `vllm-ascend-perf-hunter`
   - add `vllm-ascend-graph-analyzer` when graph/capture/replay/scheduler symptoms appear
   - add `vllm-ascend-test-matrix-planner` when the current evidence is not enough to isolate the bottleneck
   - add `repo-state-auditor` when the issue is explicitly a change-induced regression
5. Use `_shared` knowledge to anchor the diagnosis only after the evidence threshold is met:
   - `design_analysis_index.json` for architecture surfaces
   - `imported_knowledge_report.json` for evidence-backed categories and gaps
   - `code-knowledge-map.md` for code-path jumps
6. Converge on one primary bottleneck class. If the data is ambiguous, return ranked suspects rather than overcommitting.
7. End with a minimal experiment loop that can prove or falsify the current hypothesis.

## Output Contract

Always return these sections:

- `Perf question`
- `Primary classification`
- `Evidence inventory`
- `Observed bottleneck`
- `Ranked suspects`
- `Likely code surfaces`
- `Next experiment plan`
- `Acceptance signal`
- `Open gaps`

## Guardrails

- Do not confuse observability overhead with the underlying service bottleneck.
- Do not attribute a regression to kernels or operators when the queueing or scheduling evidence says otherwise.
- Do not recommend wide parameter sweeps before isolating one control variable.
- If only one noisy run exists, treat the result as suggestive, not conclusive.
- If profiling coverage is incomplete, say what is missing before giving a high-confidence tuning recommendation.
- Do not load heavyweight shared indexes for a one-line profiling request with no artifacts or metrics.
