---
topic_id: vllm.arg.max_num_batched_tokens
canonical_term: --max-num-batched-tokens
topic_kind: parameter
---

# --max-num-batched-tokens

## Core

- topic_id: `vllm.arg.max_num_batched_tokens`
- canonical_term: `--max-num-batched-tokens`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `throughput_tuning`
- status/confidence: `aligned` / `0.95`
- source: `code` / source_tags: code
- semantics: 调度和批处理参数调优，目标提升吞吐。
- aliases: `--max-num-batched-tokens`, `max-num-batched-tokens`, `max_num_batched_tokens`, `max num batched tokens`, `maxnumbatchedtokens`, `throughput_tuning`, `throughput tuning`, `throughput-tuning`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `throughput_tuning` 查看稳定原理。

## Deployment View

- default_behavior: 默认 2048（部署侧常按场景覆盖）。
- value_shape: `numeric`
- accepted_values: int >= 1
- constraints: 必须 >= max_num_seqs; 当关闭 chunked prefill 时，必须 >= max_model_len
- combo_effects: 与 max_num_seqs、max_model_len 共同决定调度上限与排队行为; 在 FlashComm1 + PCP 场景可能被对齐为 tp_size*pcp_size 的倍数

## Development View

- definition_ref: vllm/engine/arg_utils.py:1105
- read_ref: vllm/vllm/benchmarks/sweep/plot.py:580, vllm/vllm/benchmarks/sweep/plot.py:582, vllm/vllm/compilation/passes/fusion/allreduce_rms_fusion.py:736
- effect_ref: vllm/vllm/config/scheduler.py:258, vllm/vllm/config/scheduler.py:265, vllm/vllm/engine/arg_utils.py:302
- web_refs: 5

## Details/Edge Cases

- failure_modes: TTFT/TPOT 退化; OOM
- value_failure_signals: ValueError: max_num_batched_tokens must be >= max_num_seqs; ValueError: smaller than max_model_len when chunked prefill disabled
- recommendation: 按 TTFT/TPOT/吞吐三指标联合调参。
- updated_at: 2026-03-06
