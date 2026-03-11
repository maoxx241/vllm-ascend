# Graph Signals

## Common Signals

| Signal | Interpretation |
| --- | --- |
| eager mode is faster or more stable than graph mode | graph capture or replay overhead is plausible |
| shape-sensitive workloads regress while fixed-shape workloads do not | batch invariance or shape drift issue |
| capture time is large but replay is healthy | warmup/capture cost dominates, not steady-state |
| replay remains slow even after warmup | graph benefit is not materializing; inspect kernels, shapes, or scheduling interactions |
| logs mention capture/replay mismatch, compile fallback, or graph disable path | graph-specific diagnosis is mandatory |

## High-Value Knowledge Entries

- `feature:acl_graph`
- `feature:batch_invariant`
- `feature:dynamic_batch_scheduler`
- `config:xlite_graph_config`
- `api:ACLGraphWrapper`
