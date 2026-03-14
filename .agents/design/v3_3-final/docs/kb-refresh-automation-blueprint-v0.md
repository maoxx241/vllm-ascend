# KB Refresh Automation Blueprint v0

## Goal

Provide a single engineered path so an agent can refresh the typed KB from the current repository state, while preserving safety against semantic drift.

The refresh pipeline should work in three trigger modes:

1. **Manual agent trigger**: user says `更新知识库`.
2. **Commit trigger**: run on target branch HEAD or on a PR merge candidate.
3. **Nightly sweep**: rebuild against latest default branch and compare to last successful baseline.

## Core Principle

Do not let the agent directly mutate production selector/runtime behavior during refresh.

Use a staged pipeline:

\[
K_{refresh} = K_{static} \cup K_{import} \cup K_{runtime}
\]

Where:

- `K_static`: repo-static control surfaces, AST guards, docs/examples/tests.
- `K_import`: class shapes and import surfaces from the target environment.
- `K_runtime`: real runtime probes from Ascend environments.

Only `K_static` and `K_import` are mandatory for every refresh. `K_runtime` is opportunistic unless an attached Ascend pool exists.

## Recommended Architecture

### Loop A: Always-on static refresh
Runs on every target commit.

Inputs:
- repo HEAD or explicit ref
- previous successful baseline manifest

Steps:
1. checkout target ref in isolated worktree
2. run control-surface collectors
3. run AST-aware branch tracing
4. run HF trait extraction
5. run role-topology extraction
6. compile evidence catalog
7. compile typed KB tables
8. compile version provenance tables
9. compile constraint predicates
10. compile selector/runtime binding candidates
11. run dry-run fixtures
12. run shadow seam regression
13. emit diff report against previous baseline
14. write refresh manifest

Outputs:
- `root_inventory.jsonl`
- `evidence_catalog*.jsonl`
- `typed_kb_tables*.jsonl`
- `version_provenance_tables*.jsonl`
- `compiled_constraint_predicates*.jsonl`
- `selector_binding_candidates*.jsonl`
- `merge_readiness_gate*.md`
- `refresh_diff_report.md`
- `refresh_manifest.json`

### Loop B: Optional import-surface refresh
Runs whenever a Python environment matching the target branch is available.

Inputs:
- same repo ref
- target Python environment

Steps:
1. run import-surface probe
2. compare class shapes / defaults to previous import baseline
3. append `runtime_import` provenance rows
4. regenerate typed tables impacted by shape drift

Outputs:
- `import_surfaces.json`
- `import_surface_diff_report.md`

### Loop C: Runtime breadth refresh
Runs on attached Ascend pool or when runtime JSONs are supplied.

Inputs:
- runtime tuples / probes from A2/A3 variants

Steps:
1. normalize runtime samples
2. update hardware family and variant matrices
3. regenerate runtime breadth report
4. mark breadth caveat deltas

Outputs:
- `runtime_tuple.json`
- `runtime_sample_breadth*.md`
- `hardware_taxonomy_seed*.yaml`

## Refresh State Machine

### State 0: `collect`
Collect all available evidence.

### State 1: `compile`
Compile evidence into typed intermediate tables.

### State 2: `validate`
Run dry-run fixtures and shadow regression.

### State 3: `classify`
Classify the delta:
- additive surface change
- semantic drift
- runtime breadth change
- blocking regression

### State 4: `publish`
Publish one of:
- `green`: artifact refresh complete, no blocking regression
- `yellow`: refresh complete, manual review needed
- `red`: refresh failed or semantic drift is blocking

## Safe Defaults

The refresh pipeline should preserve these invariants:

\[
primary\_decision_{after\ refresh} = primary\_decision_{before\ refresh}
\]

This remains true because:
- adapter is shadow-only
- feature gate stays off by default
- refresh updates artifacts and shadow envelopes first

## Drift Classification

### Low risk
- new CLI flag
- new env var
- new config field
- new additional-config subkey

Action:
- add to root inventory
- mark unclassified or unsupported until compiled

### Medium risk
- class shape changed
- default changed
- docs/code mismatch
- import path changed

Action:
- generate provenance conflict rows
- recompile typed tables
- require dry-run + shadow pass

### High risk
- existing field keeps the same name but changes semantics
- new hidden coupling across runtime/backend/model trait
- predicate target meaning changes

Action:
- manual review required
- keep shadow-only
- do not auto-promote to primary decision path

## Branching Model

Recommended branch names:
- `kb-refresh/<date>-<shortsha>`
- `kb-refresh/nightly/<date>`
- `kb-refresh/manual/<ticket-or-user>`

The pipeline should never rewrite the current working branch in place.

## Agent Contract

The agent-facing contract should be one command:

```text
更新知识库 [optional ref]
```

Expected behavior:
1. resolve target ref, default `HEAD`
2. create isolated worktree or refresh branch
3. run Loop A
4. run Loop B if Python environment is present
5. run Loop C if Ascend runtime inputs are available
6. return:
   - refresh status
   - delta summary
   - blocking items
   - artifact bundle path
   - suggested next action

## What should be fully automatic

These parts are safe to automate on every commit:
- static collectors
- typed-table compilation
- provenance diff
- dry-run fixtures
- shadow regression
- branch/artifact generation

## What should stay conditionally automatic

These parts should run only when resources are available:
- import-surface probing in target env
- A2/A3 runtime breadth probing
- variant expansion beyond current samples

## Minimal Acceptance Gates

A refresh is considered successful when:

\[
G = G_{compile} \land G_{fixture} \land G_{shadow}
\]

Where:
- `G_compile`: all compile stages finish
- `G_fixture`: dry-run fixtures pass
- `G_shadow`: shadow seam regression passes

Runtime breadth is currently a caveat, not a hard blocker for artifact refresh.

## Recommendation for your project

The best practical setup is:

1. one repo-local orchestrator script
2. one CI workflow for commit/nightly refresh
3. one agent command that calls the same orchestrator
4. one refresh manifest stored with outputs
5. one policy file that controls whether gate stays shadow-only

That gives you a single refresh path rather than separate "manual" and "agent" logic.
