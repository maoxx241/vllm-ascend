---
topic_id: vllm.arg.cp_kv_cache_interleave_size
canonical_term: --cp-kv-cache-interleave-size
topic_kind: parameter
---

# --cp-kv-cache-interleave-size

## Core

- topic_id: `vllm.arg.cp_kv_cache_interleave_size`
- canonical_term: `--cp-kv-cache-interleave-size`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `context_parallel`
- status/confidence: `needs_manual_review` / `0.86`
- semantics: 将长上下文处理拆分到多个并行单元，降低单卡压力。
- aliases: `--cp-kv-cache-interleave-size`, `cp-kv-cache-interleave-size`, `cp_kv_cache_interleave_size`, `cp kv cache interleave size`, `cpkvcacheinterleavesize`, `context_parallel`, `context parallel`, `context-parallel`

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

- definition_ref: vllm/engine/arg_utils.py:810
- read_ref: vllm/vllm/config/parallel.py:256, vllm/vllm/config/parallel.py:260, vllm/vllm/config/parallel.py:271
- effect_ref: vllm-ascend/vllm_ascend/platform.py:395
- web_refs: 7

## Details/Edge Cases

- failure_modes: KV 传输配置错误; 时延反而变高
- value_failure_signals: KV 传输配置错误; 时延反而变高
- recommendation: 优先在高并发长上下文场景启用并做 A/B。
- updated_at: 2026-03-05
