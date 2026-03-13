# vllm-ascend-assistant

The only public entry for the v3.3 runtime.

Use `runtime.py` first. Do not grep raw docs first.

Workflow:
- Build a `RawRequest`.
- Call `vllm_ascend_assistant` from `runtime.py`.
- If the result is `direct_answer`, stop.
- If the family is `deployment_execution`, continue with `deployment-intake`
  and then the deployment atomic skills instead of answering from raw docs.
- Stable skills should answer from runtime objects and capsule output, not by
  reading raw sqlite or fabricating missing launch paths.
- Hardware topology facts must come from runtime normalization and capsule
  output, not from ad hoc rules embedded in this skill text.
