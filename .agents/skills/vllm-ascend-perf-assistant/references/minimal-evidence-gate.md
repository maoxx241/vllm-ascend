# Minimal Evidence Gate

Use this file before loading heavy shared knowledge.

## Evidence Threshold

Treat the request as low-evidence if it has fewer than three of these:

- baseline and current throughput or latency numbers
- profiling artifact paths
- queue vs execute timing signal
- graph/eager comparison
- topology or parallelism info
- model and quantization info
- clear reproduction shape or request mix

## Low-Evidence Behavior

If the request is low-evidence:

- do not load heavyweight indexes
- do not jump into code-surface analysis
- do not speculate about kernels or operators
- return a collection plan first

## Heavy Files

Load these only after the evidence threshold is met:

- `../_shared/knowledge-governance/generated/design_analysis_index.json`
- `../_shared/knowledge-governance/generated/imported_knowledge_manifest.json`
- `../_shared/knowledge-governance/generated/task_skill_index.json`

`imported_knowledge_report.json` is small enough for summary use.

## Default First Response

When the request is too thin, return only:

- `Perf question`
- `Primary classification`
- `Evidence inventory`
- `Open gaps`
- `Next experiment plan`
