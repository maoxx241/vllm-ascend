---
name: vllm-ascend-deployment
description: Open-world deployment synthesis for vLLM-Ascend. Use this after the request is normalized and self-acquire has already happened. This skill must combine repo evidence, local source evidence, user-only facts, and hard resource constraints to produce a deployment bundle. It must classify into needs_alignment, exact_verified, compatible, candidate, or blocked.* and must not emit scripts for blocked results.
---

# vllm-ascend-deployment

This skill synthesizes deployment bundles. It is not a lookup table and not a closed-world KB reader.

## Iron laws

1. Unlisted is not unsupported.
2. Unverified is not impossible.
3. Self-acquire first. Ask only user-only blocker questions.
4. Support for a model family and support for a feature do not automatically imply support for every combination.
5. Scripts are emitted only for non-blocked results.

## Required output classes

- `needs_alignment`
- `exact_verified`
- `compatible`
- `candidate`
- `blocked.identity`
- `blocked.resource`
- `blocked.hard_negative`
- `blocked.conflict`

## Decision loop

1. Build the scenario object:
   - model
   - hardware
   - input length distribution
   - SLA / TPOT
   - deployment form
2. Self-acquire evidence from repo docs, nightly configs, and local source.
3. Ask only user-only blockers.
4. If blocked by hard negatives or resource impossibility, stop.
5. Otherwise synthesize a deployment bundle.

## Question gate rules

Ask the user if any of these facts are still missing and they matter:

- exact weight path or remote model identifier
- quantization status if it changes the command
- A2 card count
- average input length and average output length
- max context length
- TPOT / SLA
- whether an unusual model name is a typo or a real custom model
- whether the user already has a quantized weight artifact when the request names a quantized model variant

Do not ask questions that the repo or local source can answer.

## Normal feature policy

Do not label these as experimental by default:

- MTP
- FULL_DECODE_ONLY / full graph style compilation
- quantized deployment when the user already has a matching quantized weight artifact

These are normal deployment choices unless the specific model / weight / hardware combination is genuinely novel or contradicted by evidence.

## Bundle artifacts

For non-blocked results, emit:

- `result.json`
- `decision_report.md`
- `validation_checklist.md`
- `scripts/launch_primary.sh`
- optional `scripts/launch_alternative.sh`

For blocked results, emit only:

- `result.json`
- `decision_report.md`
- `validation_checklist.md`

## Allowed helper

```bash
python tools/vas_deployment_open_world/cli.py synthesize \
  --repo-root . \
  --request "<user request>" \
  --out .vas/cases/<case_id>
```
