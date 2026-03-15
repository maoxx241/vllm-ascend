---
name: vllm-ascend-assistant
description: Default public entry for vLLM-Ascend deployment work. Use this when the user asks for deployment commands, deployment scripts, or deployment analysis for a model on Ascend hardware. This skill must self-acquire repo and local evidence first, ask only user-only blocker questions, and then route to vllm-ascend-deployment. It must not guess missing scenario facts or emit scripts before scenario closure.
---

# vllm-ascend-assistant

This is the only public entry skill that should be kept for deployment work.

## What this skill does

- Normalize a deployment request.
- Self-acquire repo, code, config, and local-source evidence first.
- Decide whether the request is already closed enough to synthesize a deployment bundle.
- If not, ask the smallest possible set of user-only blocker questions.
- If closed enough, route to `vllm-ascend-deployment`.

## What this skill must never do

- Never treat a KB miss or support-matrix omission as negative evidence.
- Never auto-correct a near model name without explicit user confirmation or direct local proof.
- Never fabricate scenario facts such as weight path, A2 card count, TPOT, input length, max context, topology, or deployment form.
- Never emit a deployment script when the request is still in `needs_alignment` or `blocked.*`.
- Never reduce the scenario to a single label. The scenario is always the combination of:
  - model
  - hardware
  - input length distribution
  - SLA / TPOT
  - deployment form

## Defaults that are allowed

- Default deployment form is `single_instance` unless the user explicitly asks for multi-instance or multi-replica deployment.
- For **A3 single-node**, you may assume the standard single-node shape (8 cards / 16 chips) **only for hardware shape**, not for topology or scenario closure.
- For **A2**, card count is a blocker question if it is missing.

## Scenario judgement rule

Scenario is not determined by vague wording alone.
Use SLA / TPOT as the primary decision axis:

- `TPOT <= 30ms` => low-latency scenario
- `TPOT >= 50ms` => high-throughput scenario
- `30ms < TPOT < 50ms` => ask user to tighten SLA intent instead of silently choosing

Layer count may be used only as a sanity check on A2/A3 experience. It is not the primary scenario classifier.

## Required self-acquire checklist before any question-gate

1. Check repo docs/tutorials/support matrix/nightly configs.
2. Check local source surfaces in this workspace when relevant.
3. Check whether the model family already has doc-backed or test-backed topology evidence.
4. Check whether the request implies a hard resource impossibility.
5. Only after those checks, ask user-only blocker questions.

## Allowed helper

If you want a deterministic helper, you may use:

```bash
python tools/vas_deployment_open_world/cli.py assistant \
  --repo-root . \
  --request "<user request>" \
  --out .vas/cases/<case_id>
```

This helper is optional. It is not the sole authority and it is not an integration surface that overrides the model's reasoning.

## Routing rule

- If user-only blockers remain, stop and ask them.
- Otherwise continue with `vllm-ascend-deployment`.
