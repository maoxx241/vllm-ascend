---
name: vllm-ascend-perf-hunter
description: Analyze vLLM Ascend profiling artifacts, benchmark summaries, and service-level metrics to identify primary bottlenecks and hotspot categories. Use when performance evidence already exists and needs structured interpretation.
---

# vLLM Ascend Perf Hunter (A6)

## Purpose

Read offline profiling and benchmark artifacts, then classify the dominant bottleneck with evidence and a short suspect list.

## Read Order

1. `../_shared/code-knowledge-map.md`
2. `../_shared/knowledge-governance/generated/imported_knowledge_manifest.json`
3. `../_shared/knowledge-governance/generated/design_analysis_index.json`
4. `../_shared/deployment-config/references/global-parameter-feature-map.md`
5. `references/perf-inputs.md`
6. `references/bottleneck-patterns.md`

## Supported Inputs

- `msserviceprofiler` outputs such as `service_summary.csv`, `request_summary.csv`, `batch_summary.csv`, `kvcache.csv`, `chrome_tracing.json`
- `torch.profiler` traces or summaries
- `nsys stats` summaries
- `vllm bench` or custom benchmark logs
- run config snapshots: CLI flags, env vars, topology, model info

## Workflow

1. Inventory the artifact types and note missing pieces.
2. Extract the highest-signal metrics first:
   - throughput, TTFT, TPOT, tail latency
   - queueing vs execute time
   - batch scheduling behavior
   - KV/cache pressure and memory churn
   - graph capture/replay cost if present
3. Classify the dominant bottleneck using `references/bottleneck-patterns.md`.
4. Map the bottleneck to likely code surfaces through `_shared` manifests and `code-knowledge-map.md`.
5. Return one primary bottleneck plus up to three secondary suspects.
6. If the artifacts are insufficient to separate suspects, say so and hand off to `test-matrix-planner`.

## Output Contract

- `Artifacts read`
- `Primary bottleneck`
- `Key evidence`
- `Secondary suspects`
- `Likely code surfaces`
- `Missing evidence`

## Guardrails

- Prefer service-level timing decomposition over raw intuition.
- Do not treat one hot kernel name as the whole bottleneck without correlating it to end-to-end impact.
- Distinguish warmup/capture cost from steady-state serving cost.
- If profiling itself is likely distorting the workload, call that out explicitly.
