# Version provenance tables v0

This document defines the first typed representation for runtime/import-version drift.

## Problem

The current KB work must accept evidence from:

- repo-static extraction
- runtime probe data
- runtime import surfaces
- operator-supplied notes

These sources can legitimately disagree when the observed runtime is not using the
same vllm / vllm-ascend checkout as the workspace being analyzed. Treating those
observations as one merged truth would silently corrupt the KB.

## Table intent

The version provenance pass preserves disagreement instead of resolving it.

### `runtime_stack_observations`
Authoritative for runtime-observed software stack and hardware-shape facts.

### `import_surface_observations`
Authoritative for class/env/CLI surfaces that can be imported in the observed runtime.

### `import_surface_shapes`
Authoritative for per-class field-shape drift across observed runtimes.

### `version_conflict_rows`
Not a fact table. This is a *reconciliation policy* table. Each row records:

- divergence dimension
- severity
- per-sample values
- policy hint

## Current policy

- Partition by runtime sample and provenance.
- Do not collapse divergent class shapes into one canonical config surface yet.
- Prefer runtime probe data over wheel tags for NPU capability.
- Treat CLI-help failures as non-authoritative unless corroborated elsewhere.

## Why this matters

The observed A2 and A3 environments already diverge in:

- torch / torch_npu versions
- import paths
- git reliability of nested repos
- `VllmConfig` / `ParallelConfig` / `ModelConfig` field shapes
- CLI availability / failure patterns

These are exactly the kinds of drifts that must remain explicit until we have a
branch-aware multiversion policy compiled into the runtime-facing KB.
