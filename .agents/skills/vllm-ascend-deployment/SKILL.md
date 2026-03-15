# vllm-ascend-deployment

Open-world deployment synthesis for vLLM-Ascend.

This skill is not a lookup table and not a closed-world KB reader.
It must:
- self-acquire from repo, local source, upstream mirror, docs, and recipes
- minimize user questions to user-only blocker facts
- classify results into:
  - exact_verified
  - exact_derived
  - compatible
  - candidate
  - blocked.*
- emit bundle artifacts instead of just a single command

Bundle artifacts:
- result.json
- decision_report.md
- validation_checklist.md
- scripts/*.sh for non-blocked results

Guard rules:
- no scripts for blocked results
- no automatic typo correction without confirmation or local proof
- no closed-world reasoning from support-matrix omissions
- generic model support + generic quant support != verified combination
