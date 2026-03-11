# Debug Taxonomy

Use this file to force one primary failure class before deeper analysis.

## Failure Classes

| Class | Typical signals | Default chain | Do not confuse with |
| --- | --- | --- | --- |
| `environment` | missing `.so`, import failure, bad `ASCEND_HOME_PATH`, bad `SOC_VERSION`, runtime init failure before model load | `compatibility-checker` -> `env-bootstrap` -> `log-analyzer` | request-time model/runtime bugs |
| `distributed_runtime` | HCCL init, rank mismatch, communicator timeout, worker bootstrap failure in distributed mode | `log-analyzer` -> `crash-rooter` -> `compatibility-checker` | pure model registration failures |
| `crash_or_oom` | process exit, segfault, assert, Python exception, worker death, CUDA/NPU/OOM signature | `crash-rooter` -> `log-analyzer` | slowdowns without a hard failure |
| `graph_or_compile` | ACL graph capture failure, replay mismatch, shape drift, compile/dynamo issues | `graph-analyzer` -> `log-analyzer` | generic operator correctness bugs |
| `operator_or_numeric` | wrong output, kernel mismatch, precision drift, unsupported operator path | `log-analyzer` -> `graph-analyzer` -> `precision-validator` | environment bootstrap issues |
| `memory_or_kv` | memory growth, KV-cache corruption, prefix-cache anomaly, paged-attention / reshape-and-cache issues | `crash-rooter` -> `log-analyzer` -> `attention-kv-designer` | pure throughput regressions |
| `performance_regression` | throughput drop, tail-latency spike, regression after change while correctness is intact | `perf-hunter` -> `test-matrix-planner` -> `repo-state-auditor` | crash/OOM events |
| `unknown_or_log_only` | only partial logs, unclear symptom, no stack, no repro | `log-analyzer` first, then reclassify | pretending certainty too early |

## Minimal Evidence Bundle

Collect these before diagnosis:

- failing command, API request, or reproduction steps
- failure phase: `bootstrap | startup | first_request | steady_state | regression`
- first bad log line
- stack trace or exception signature
- model name, quantization mode, parallelism layout, graph/eager mode
- critical env vars and CLI flags
- recent code/config/version delta

## Escalation Rules

- If you only have logs, start with `unknown_or_log_only`.
- If both crash and graph symptoms appear, classify as `graph_or_compile` only when the crash is secondary to graph capture/replay.
- If both env drift and request crash appear, classify as `environment` only when the request never reaches a valid model/runtime path.
