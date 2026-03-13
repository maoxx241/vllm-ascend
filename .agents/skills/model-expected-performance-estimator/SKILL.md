# model-expected-performance-estimator

Atomic performance skill that returns an expected TTFT/throughput/memory
envelope instead of measured results.

Use runtime and capsule output first. Do not invent a measured single point or
fabricate a baseline from raw docs. If runtime has not selected a stable
artifact path or topology strategy, return a conditional envelope and keep the
route unresolved.
