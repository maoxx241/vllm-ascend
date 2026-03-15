# Minimal agent surface for vLLM-Ascend

This package intentionally replaces the previous `.agents` subtree with a minimal, clean surface.

Only two public skills remain:

- `vllm-ascend-assistant`: default public entry for deployment requests
- `vllm-ascend-deployment`: open-world deployment synthesis

Everything else under the old `.agents` tree is treated as legacy and removed during merge.
