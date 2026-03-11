# Entry Routing

Use this as a cheap first-hop gate before loading deployment-heavy knowledge.

## Hard Rule

If the request is primarily about any of these, stop and hand off to `vllm-ascend-developer-assistant`:

- profiling or performance analysis
- debugging, crash, OOM, logs, replay mismatch
- design or architecture analysis
- model adaptation
- upstream sync
- release analysis
- operator or kernel development

## Continue In Deployment Assistant Only For

- deployment intent
- environment bootstrap
- deployment-side feature compatibility and parameter explanation

## Example

- `我要分析profiling，然后用kimik2模型看看瓶颈`
  - classify as `performance_analysis`
  - route to `vllm-ascend-developer-assistant`
  - do not load `ai-foundation` or deployment KB first
