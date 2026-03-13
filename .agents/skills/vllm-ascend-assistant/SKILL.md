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
- For A3 deployment requests, physical cards are not logical NPUs:
  `1 card = 2 logical NPUs`. Preserve that mapping before inferring TP.
