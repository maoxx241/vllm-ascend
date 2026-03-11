# Perf Inputs

## Preferred Evidence Order

1. service-level summaries:
   - `service_summary.csv`
   - `request_summary.csv`
   - `batch_summary.csv`
2. trace-level evidence:
   - `chrome_tracing.json`
   - Perfetto / torch profiler / nsys summaries
3. workload context:
   - model, quantization, sequence lengths, request mix
   - graph/eager mode
   - topology and parallelism settings
4. run config:
   - profiling flags
   - important env vars
   - recent code or config deltas

## Useful Repo Docs

- `vllm-ascend/docs/source/developer_guide/performance_and_debug/service_profiling_guide.md`
- `vllm-ascend/docs/source/developer_guide/performance_and_debug/performance_benchmark.md`
- `vllm/docs/contributing/profiling.md`

## If Inputs Are Weak

Return a collection request instead of forcing analysis:

- capture one baseline and one regression run
- preserve the exact launch flags
- preserve request shape and concurrency
- include one service-level summary and one trace artifact
