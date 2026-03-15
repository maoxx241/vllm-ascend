# vllm-ascend-assistant

Default public entry for agent-driven vLLM-Ascend work.

Current acceptance scope in this package:
- full deployment bootstrap and routing
- open-world self-acquire before question-gate
- case workspace + deployment bundle writeback

Do this first for deployment-style requests:
1. Normalize the request.
2. Self-acquire repo/code/doc evidence.
3. Only ask user-only blocker questions.
4. Route to deployment synthesis.
5. Emit a bundle: result.json, decision_report.md, validation_checklist.md, shell scripts if applicable.

Do **not**:
- treat KB miss as negative evidence
- auto-correct near model names without user confirmation or direct local evidence
- fabricate hardware, card count, weight path, or topology
- emit shell scripts for blocked results

`runtime.py` is the integration surface.
