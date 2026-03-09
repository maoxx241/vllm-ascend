# NPU `causal_conv1d_update` Fix Notes

This note records the NPU-side fixes for `causal_conv1d_update` in `vllm_ascend`.

## Background

The previous implementation diverged from upstream `vllm` in two important ways:

- The wrapper rewrote layouts with `transpose(...).contiguous()`, which hid stride bugs and broke the public tensor contract.
- The Triton kernel forced history and token loads to `fp16`, which introduced avoidable precision loss for `bf16` and `fp32` inputs.

The updated implementation restores the upstream tensor semantics and keeps the kernel stride-aware end to end.

## Scope

The fix targets the single operator only:

- correctness for decode / batch-gather / speculative decode / varlen paths
- support for non-contiguous tensors with valid strides
- vector-core-aware launch parameter selection

Model-level regression is intentionally out of scope for this change set.

## Implementation

The updated operator lives in [vllm_ascend/ops/triton/mamba/causal_conv1d.py](/mnt/c/Users/maoxx241/code/vllm_workspace/vllm-ascend/vllm_ascend/ops/triton/mamba/causal_conv1d.py).

Key changes:

- remove wrapper-side `transpose(...).contiguous()` rewrites for `weight` and `conv_state`
- parse `x`, `conv_state`, `weight`, and `out` using their real strides
- keep `out = x` behavior for decode inputs instead of silently materializing a new contiguous output
- use source element dtype for history/token loads, accumulate in `fp32`, and cast back to the original output dtype
- replace the hard-coded `CORE_HINT = 40` heuristic with runtime vector-core detection via `init_device_properties_triton()` and `get_vectorcore_num()`
- expose the launch heuristic as `_pick_causal_conv1d_update_launch_params(...)` so 20-core and 24-core scheduling can be unit-tested directly

## Tests

The operator tests live in [test_causal_conv1d.py](/mnt/c/Users/maoxx241/code/vllm_workspace/vllm-ascend/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_causal_conv1d.py).

Added coverage:

- strided `x`, `conv_state`, and `weight` views
- `fp32` and `bf16`
- launch heuristic checks for 20 and 24 vector cores

Recommended validation commands:

```bash
cd /workspace/vllm_workspace/vllm-ascend
pytest -v -s tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_causal_conv1d.py
```

```bash
cd /workspace/vllm_workspace/vllm
pytest -v -s tests/kernels/mamba/test_causal_conv1d.py
```

## Performance Notes

The fast path should remain at least performance-neutral for the common contiguous decode case. The general stride path is allowed to be slower than the fast path, but it must remain correct and must not regress the common contiguous case.

When benchmarking on 173, record at least:

- a contiguous decode case
- a non-contiguous decode case
- batch, dim, width, seqlen, dtype, and measured latency

## References

- upstream `vllm` causal conv1d implementation in `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`
- [Triton Ascend repository](https://gitcode.com/Ascend/triton-ascend)
