# NPU `causal_conv1d_update` Fix Notes

This note records the NPU-side correctness and performance work for `causal_conv1d_update` in `vllm_ascend`.

## Background

The old NPU path had three separate problems:

- the wrapper rewrote layouts with `transpose(...).contiguous()`, which hid stride bugs and broke the public tensor contract
- the Triton kernel forced history and token loads to `fp16`, which introduced avoidable precision loss for `bf16` and `fp32`
- decode, update, and MTP all shared one generic runtime-heavy kernel, which was correct only after additional fixes and still far from the latency target

## Implementation

The implementation lives in [causal_conv1d.py](/mnt/c/Users/maoxx241/code/vllm_workspace/vllm-ascend/vllm_ascend/ops/triton/mamba/causal_conv1d.py).

The operator now has two layers:

- a generic Triton fallback that keeps full stride-aware correctness for decode, update, speculative decode, batch gather, and varlen
- specialized `bf16 + width=4` fast paths for the dominant performance-sensitive cases

Fast-path dispatch is explicit:

- `decode_contig_s1_bf16_w4`
- `decode_stride_s1_bf16_w4`
- `update_contig_s3_bf16_w4`
- `update_stride_s3_bf16_w4`
- `mtp_contig_k3_bf16_w4`
- `mtp_stride_k3_bf16_w4`

Key implementation changes:

- remove wrapper-side `transpose(...).contiguous()` rewrites for `weight` and `conv_state`
- parse `x`, `conv_state`, `weight`, and `out` using their real strides
- keep `out = x` behavior instead of silently materializing a new output tensor
- use source element dtype for loads, accumulate in `fp32`, and cast back to the original output dtype
- add an internal read-only weight prepack cache so fast paths always load packed `[width, dim]` weights
- replace the hard-coded `CORE_HINT = 40` heuristic with runtime vector-core detection and path-specific launch selection
- expose both generic and fast-path launch helpers so 20-core and 24-core scheduling can be unit-tested directly

## Tests

The main operator test file is [test_causal_conv1d.py](/mnt/c/Users/maoxx241/code/vllm_workspace/vllm-ascend/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_causal_conv1d.py).

Coverage now includes:

- contiguous and non-contiguous decode (`seqlen=1`)
- contiguous and non-contiguous update (`seqlen=3`)
- contiguous and non-contiguous MTP (`max_query_len=4`, `num_accepted_tokens in {1,2,3,4}`)
- `bf16` and `fp32` correctness
- fast-path dispatch checks
- weight prepack cache hit and invalidation checks
- 20-core and 24-core launch parameter checks
- optional latency gates for the 6 SLA cases

Operator validation:

```bash
cd /workspace/vllm_workspace/vllm-ascend
pytest -v -s tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_causal_conv1d.py
```

Latency gate validation:

```bash
cd /workspace/vllm_workspace/vllm-ascend
VLLM_ASCEND_RUN_PERF=1 pytest -v -s tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_causal_conv1d.py -k perf_gate
```

Relevant upstream regression:

```bash
cd /workspace/vllm_workspace/vllm
pytest -v -s tests/kernels/mamba/test_causal_conv1d.py
```

```bash
cd /workspace/vllm_workspace/vllm
pytest -v -s tests/v1/e2e/test_mamba_prefix_cache.py -k test_mamba_prefix_cache
```

```bash
cd /workspace/vllm_workspace/vllm
pytest -v -s tests/v1/e2e/test_spec_decode.py -k test_mtp_correctness
```

## Benchmarking

The microbenchmark helper is [benchmark_causal_conv1d_update.py](/mnt/c/Users/maoxx241/code/vllm_workspace/vllm-ascend/benchmarks/ops/benchmark_causal_conv1d_update.py).

It prints:

- selected fast path
- selected launch config
- mean / p50 / p95 latency in microseconds

Example:

```bash
cd /workspace/vllm_workspace/vllm-ascend
python3 benchmarks/ops/benchmark_causal_conv1d_update.py --case all --batch 64 --dim 4096
```

The performance target for the fast paths is `<30us` on the following cases:

- contiguous decode
- non-contiguous decode
- contiguous update
- non-contiguous update
- contiguous MTP
- non-contiguous MTP

## References

- upstream `vllm` causal conv1d implementation in `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`
- [Triton Ascend repository](https://gitcode.com/Ascend/triton-ascend)
