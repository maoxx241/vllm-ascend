# Acceptance Matrix

## A. Contract Layer

| ID | Case | Expected |
| --- | --- | --- |
| A1 | load all schemas | success |
| A2 | validate all examples | success |
| A3 | init empty sqlite with `merged_pack.sql` | success |
| A4 | schema vs docs key names | no mismatch |
| A5 | invalid intake plan with `max_deep_refs = 1` | rejected |
| A6 | invalid reroute card without `reroute` | rejected |
| A7 | invalid confirmation seed (`required=true`, `status=not_needed`) | rejected |
| A8 | invalid intake-origin atomic plan with `spec_plan_workflow` | rejected |
| A9 | invalid continuation state with non-`full_bundle` persistence | rejected |

## B. Shared Runtime

| ID | Case | Expected |
| --- | --- | --- |
| B1 | raw request -> selector_seed/v3 | deterministic |
| B2 | pending confirmation -> Intake tries to enter Atomic | blocked |
| B3 | user_declined -> Intake emits selector_plan | blocked |
| B4 | `Public Entry` 与 Intake `direct_answer/no-query` path | no governor call |
| B5 | `query_stage` matrix enforced | no illegal budget/capsule combo |
| B6 | pending flush before second capsule | governor deny |
| B7 | duplicate plan | deduped |
| B8 | `origin_stage + query_stage + execution_mode` mismatch | rejected |
| B9 | governor stage source | derived only from `selector_plan.query_stage` |

## C. KB Runtime

| ID | Case | Expected |
| --- | --- | --- |
| C1 | resolve exact tuple | `match_level = exact` |
| C2 | resolve fallback tuple | `match_level = compatible` |
| C3 | resolve miss | `match_level = unknown` |
| C4 | build-local repo-only | success |
| C5 | pack under budget | capsule returned |
| C6 | pack over budget | smaller result or explicit miss |
| C7 | pack miss | `unknowns` non-empty |
| C8 | `model_expectation` intent | deterministic response envelope |

## D. Bundle / Persistence

| ID | Case | Expected |
| --- | --- | --- |
| D1 | first complex task turn | bundle created |
| D2 | atomic complete | progress updated before next query |
| D3 | needs_reroute result card | `reroute` + `resolution_code` + `reroute_task` + `flush_required = true` |
| D4 | continuation refresh | bundle already flushed |
| D5 | continuation state | canonical source files contain spec/plan/checklist/progress |
| D6 | continuation state persistence mode | always `full_bundle` |

## E. Canonical Routes

| ID | Case | Expected |
| --- | --- | --- |
| E1 | qwen3-next A2 baseline deployment | `deployment_execution + direct_atomic_workflow` |
| E2 | single profile breakdown | `performance_analysis + direct_atomic_workflow` |
| E3 | model expected performance envelope | `performance_analysis + direct_atomic_workflow` |
| E4 | minimal test selection from diff | `validation_strategy + direct_atomic_workflow` |
| E5 | runtime error triage (deferred P3B) | `debugging + direct_atomic_workflow` |
| E6 | multi-feature route conflict | `design_analysis + spec_plan_workflow` |
| E7 | single upstream delta | `upstream_sync + direct_atomic_workflow` |
| E8 | whole release sync | `upstream_sync + spec_plan_workflow` |
| E9 | operator capability gap | `operator_development` |

## F. Negative Cases

| ID | Case | Expected |
| --- | --- | --- |
| F1 | skill reads raw sqlite directly | fail |
| F2 | public_entry opens deep ref | fail |
| F3 | atomic expands unrelated domain | fail |
| F4 | exact miss silently treated as compatible | fail |
| F5 | code-change task with `analysis_depth = none` | fail |
| F6 | pending confirmation still enters Atomic | fail |
| F7 | user_declined still queries | fail |
| F8 | continuation_state uses `light_bundle` or `none` | fail |
| F9 | `evaluate_governor` still exposes explicit `stage` input or accepts caller override | fail |
