# Bottleneck Patterns

## Service-Level Patterns

| Pattern | Signal | Likely class |
| --- | --- | --- |
| queue dominates execute | requests wait much longer than model execution | `scheduler_or_batching` |
| capture cost dominates early iterations, steady state recovers | graph startup overhead | `graph_overhead` |
| execute time grows with memory pressure and KV activity | cache / allocator pressure | `memory_pressure` |
| utilization low with high comm time or rank skew | parallel or communication bottleneck | `parallel_or_comm` |
| hot kernels increase but end-to-end metrics barely move | local hotspot, not necessarily primary bottleneck | secondary suspect only |
| profiling on/off gap is very large | observability overhead or distorted workload | `profiling_interpretation` |

## Ranking Rules

- One primary bottleneck only.
- Secondary suspects should be capped at three.
- If evidence supports two equal explanations, say the tie explicitly and request one discriminating experiment.
