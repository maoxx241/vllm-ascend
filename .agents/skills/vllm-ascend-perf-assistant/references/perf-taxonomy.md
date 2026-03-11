# Perf Taxonomy

Use this file to force one primary performance question before deeper analysis.

## Primary Classes

| Class | Typical signals | Default chain | Avoid confusing with |
| --- | --- | --- | --- |
| `throughput_regression` | tokens/s drop, MFU drop, lower steady-state throughput | `perf-hunter` -> `test-matrix-planner` | one-off startup slowness |
| `latency_regression` | TTFT/TPOT/tail latency increase | `perf-hunter` -> `test-matrix-planner` | pure utilization issues |
| `graph_overhead` | graph capture cost, replay mismatch, eager faster than graph | `perf-hunter` -> `graph-analyzer` | generic kernel inefficiency |
| `memory_pressure` | KV/cache pressure, allocator churn, memory spikes reducing throughput | `perf-hunter` -> `test-matrix-planner` | correctness-only OOM debugging |
| `scheduler_or_batching` | queue growth, bad batch mix, decode starvation, prefill chunk effects | `perf-hunter` -> `graph-analyzer` -> `test-matrix-planner` | kernel-level bottlenecks |
| `parallel_or_comm` | TP/DP/CP imbalance, HCCL or comm overhead visible in perf traces | `perf-hunter` -> `test-matrix-planner` | environment bootstrap failures |
| `profiling_interpretation` | user already has traces/logs and needs explanation | `perf-hunter` | planning new experiments too early |

## Minimal Evidence Bundle

- baseline and regression numbers
- profiling artifact type: `msserviceprofiler | torch_profiler | nsys | benchmark_log | csv_summary`
- hardware topology and parallelism layout
- graph/eager mode
- quantization and model size
- key perf-related flags or env vars

## Escalation Rules

- If the artifact set is rich but the hypothesis is weak, escalate to `test-matrix-planner`.
- If graph mode is implicated even indirectly, include `graph-analyzer`.
- If the only data is a single wall-clock number, return a collection plan first.
