---
topic_id: vllm.arg.max_num_seqs
canonical_term: --max-num-seqs
topic_kind: parameter
---

# --max-num-seqs

## Core

- topic_id: `vllm.arg.max_num_seqs`
- canonical_term: `--max-num-seqs`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `throughput_tuning`
- status/confidence: `needs_manual_review` / `0.83`
- source: `code` / source_tags: code
- semantics: 调度和批处理参数调优，目标提升吞吐。
- aliases: `--max-num-seqs`, `max-num-seqs`, `max_num_seqs`, `max num seqs`, `maxnumseqs`, `throughput_tuning`, `throughput tuning`, `throughput-tuning`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `throughput_tuning` 查看稳定原理。

## Deployment View

- default_behavior: 默认 128（部署侧常按并发目标覆盖）。
- value_shape: `numeric`
- accepted_values: int >= 1
- constraints: 必须 <= max_num_batched_tokens
- combo_effects: 建议满足 max_num_seqs * data_parallel_size >= 实际并发目标

## Development View

- definition_ref: vllm/engine/arg_utils.py:1112
- read_ref: vllm/vllm/benchmarks/sweep/serve.py:410, vllm/vllm/config/compilation.py:582, vllm/vllm/config/compilation.py:584
- effect_ref: vllm/vllm/config/compilation.py:582, vllm/vllm/config/scheduler.py:258, vllm/vllm/config/scheduler.py:265
- web_refs: 5

## Details/Edge Cases

- failure_modes: TTFT/TPOT 退化; OOM
- value_failure_signals: ValueError: max_num_batched_tokens must be >= max_num_seqs
- recommendation: 按 TTFT/TPOT/吞吐三指标联合调参。
- updated_at: 2026-03-06
