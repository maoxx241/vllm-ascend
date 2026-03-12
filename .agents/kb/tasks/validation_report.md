# Validation Report

Ran on package: `vllm-ascend-agent-design-package-v3_3-final`

Command:

```bash
python tools/validate_design_package.py
```

Output:

```text
OK  bootstrap requirements present
OK  example -> schema  atomic-result-card.complete.json -> atomic-result-card.schema.json
OK  example -> schema  atomic-result-card.performance.expectation.complete.json -> atomic-result-card.schema.json
OK  example -> schema  atomic-result-card.performance.partial.json -> atomic-result-card.schema.json
OK  example -> schema  atomic-result-card.reroute.json -> atomic-result-card.schema.json
OK  example -> schema  atomic-result-card.validation.complete.json -> atomic-result-card.schema.json
OK  example -> schema  continuation-state.upstream-sync.json -> continuation-state.schema.json
OK  example -> schema  governor-decision.flush-required.json -> governor-decision.schema.json
OK  example -> schema  kb-pack-request.debugging.json -> kb-pack-request.schema.json
OK  example -> schema  kb-pack-request.model-expectation.json -> kb-pack-request.schema.json
OK  example -> schema  kb-pack-response.debugging.json -> kb-pack-response.schema.json
OK  example -> schema  kb-pack-response.model-expectation.json -> kb-pack-response.schema.json
OK  example -> schema  kb-resolve-result.compatible.json -> kb-resolve-result.schema.json
OK  example -> schema  selector-plan.deployment.intake.json -> selector-plan.schema.json
OK  example -> schema  selector-plan.design.spec.json -> selector-plan.schema.json
OK  example -> schema  selector-plan.performance.atomic.json -> selector-plan.schema.json
OK  example -> schema  selector-plan.performance.expectation.atomic.json -> selector-plan.schema.json
OK  example -> schema  selector-plan.validation.atomic.json -> selector-plan.schema.json
OK  example -> schema  selector-seed.adaptation.pending-confirmation.json -> selector-seed.schema.json
OK  example -> schema  selector-seed.deployment.json -> selector-seed.schema.json
OK  example -> schema  selector-seed.performance.expectation.json -> selector-seed.schema.json
OK  example -> schema  selector-seed.upstream.user-declined.json -> selector-seed.schema.json
OK  negative seed: required=true + status=not_needed rejected
OK  negative plan: intake plan with deep refs rejected
OK  negative plan: intake-origin atomic plan with spec_plan_workflow rejected
OK  negative plan: intake query_stage with atomic budget rejected
OK  negative card: reroute without payload rejected
OK  negative card: reroute without reroute_task rejected
OK  negative continuation: non-full_bundle persistence rejected
OK  contract docs/schema lint: governor stage has a single source of truth
OK  SQL init smoke test
OK  codex_backlog.yaml parse
PASS validated 21 examples + critical negative cases + contract lint + SQL smoke + backlog parse
```
