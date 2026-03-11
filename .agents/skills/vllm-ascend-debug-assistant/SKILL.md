---
name: vllm-ascend-debug-assistant
description: Route vLLM Ascend debugging tasks across logs, crashes, graph failures, environment drift, distributed runtime issues, and performance regressions. Use when a user reports startup failures, request-time crashes, HCCL/runtime errors, graph replay mismatches, OOM, memory growth, or unexplained behavior on vLLM Ascend.
---

# vLLM Ascend Debug Assistant (C3)

## Purpose

Turn a vague failure report into a deterministic debug path, then deliver a diagnosis with evidence, workaround, permanent fix direction, and verification steps.

## Read Order

1. `../_shared/INDEX.md`
2. `../_shared/task-index.md`
3. `../_shared/code-knowledge-map.md`
4. `../_shared/knowledge-governance/generated/task_skill_index.json`
5. `../_shared/knowledge-governance/generated/imported_knowledge_report.json`
6. `../_shared/knowledge-governance/generated/imported_knowledge_manifest.json`
7. `references/debug-taxonomy.md`

## When To Use

Use this skill when the primary task type is `debugging`, especially for:

- startup failures
- request-time crashes or hangs
- HCCL / runtime library / worker initialization issues
- ACL graph capture, replay, or shape drift failures
- OOM, memory growth, or KV-cache anomalies
- operator/runtime behavior mismatches
- unexplained throughput or latency regressions

## Workflow

1. Classify the failure into exactly one primary class using `references/debug-taxonomy.md`.
2. Build a minimal evidence bundle before reasoning further:
   - failing command or request
   - trigger phase: bootstrap / startup / first request / steady-state / regression
   - first bad log line
   - stack trace or crash signature
   - model, quantization, topology, and critical flags/env vars
   - recent code/config/version changes
3. Route the investigation:
   - logs present: `log-analyzer`
   - startup crash, assert, OOM, segfault, worker death: `crash-rooter`
   - graph capture, replay, shape drift, compile, or dynamo issues: `graph-analyzer`
   - version drift, environment mismatch, missing runtime libs, unsupported feature combos: `compatibility-checker`
   - suspected recent-change regression: `repo-state-auditor`
   - throughput or latency regression: `perf-hunter` and `test-matrix-planner`
4. Correlate the symptom with `_shared` knowledge:
   - use `task_skill_index.json` to find likely relevant knowledge slices
   - use `imported_knowledge_manifest.json` to pull code paths and evidence-bearing entries
   - use `code-knowledge-map.md` to jump from code/log paths to stable docs
5. Narrow to one primary root-cause hypothesis. If the evidence does not support a single root cause, return ranked hypotheses instead of pretending certainty.
6. Produce a fix plan with two levels:
   - temporary stop-loss or isolation step
   - permanent repair direction
7. End with a verification checklist that proves the issue is closed rather than hidden.

## Output Contract

Always return these sections:

- `Symptom`
- `Primary classification`
- `Evidence`
- `Root cause`
- `Immediate workaround`
- `Permanent fix direction`
- `Verification steps`
- `Open gaps`

## Guardrails

- Do not call startup success a pass if the first real request still fails.
- Do not collapse graph failures, env failures, and model-adaptation failures into one bucket.
- Do not claim a root cause without tying it to specific logs, code paths, or validated knowledge entries.
- If the evidence is weak, say `best current hypothesis`, not `root cause confirmed`.
- If a relevant knowledge entry is not `validated`, prefer local code truth and mark the confidence downgrade.
- If logs are insufficient, return a minimal repro and data-collection checklist before deeper speculation.
