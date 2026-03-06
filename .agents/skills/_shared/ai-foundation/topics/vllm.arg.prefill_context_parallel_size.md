---
topic_id: vllm.arg.prefill_context_parallel_size
canonical_term: --prefill-context-parallel-size
topic_kind: parameter
---

# --prefill-context-parallel-size

## Core

- topic_id: `vllm.arg.prefill_context_parallel_size`
- canonical_term: `--prefill-context-parallel-size`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `context_parallel`
- status/confidence: `aligned` / `0.98`
- source: `code` / source_tags: code
- semantics: 将长上下文处理拆分到多个并行单元，降低单卡压力。
- aliases: `--prefill-context-parallel-size`, `prefill-context-parallel-size`, `prefill_context_parallel_size`, `prefill context parallel size`, `prefillcontextparallelsize`, `context_parallel`, `context parallel`, `context-parallel`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `context_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 低卡数下收益低且配置复杂
- combo_effects: N/A

## Development View

- definition_ref: vllm/engine/arg_utils.py:814
- read_ref: vllm/vllm/config/parallel.py:101, vllm/vllm/config/parallel.py:552, vllm/vllm/engine/arg_utils.py:396
- effect_ref: vllm/vllm/platforms/rocm.py:512, vllm-ascend/vllm_ascend/ascend_config.py:71, vllm-ascend/vllm_ascend/attention/utils.py:61
- web_refs: 6

## Details/Edge Cases

- failure_modes: KV 传输配置错误; 时延反而变高
- value_failure_signals: KV 传输配置错误; 时延反而变高
- recommendation: 优先在高并发长上下文场景启用并做 A/B。
- updated_at: 2026-03-06
