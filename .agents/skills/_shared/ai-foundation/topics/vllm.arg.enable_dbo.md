---
topic_id: vllm.arg.enable_dbo
canonical_term: --enable-dbo
topic_kind: parameter
---

# --enable-dbo

## Core

- topic_id: `vllm.arg.enable_dbo`
- canonical_term: `--enable-dbo`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `throughput_tuning`
- status/confidence: `aligned` / `0.88`
- source: `code` / source_tags: code
- semantics: 调度和批处理参数调优，目标提升吞吐。
- aliases: `--enable-dbo`, `enable-dbo`, `enable_dbo`, `enable dbo`, `enabledbo`, `throughput_tuning`, `throughput tuning`, `throughput-tuning`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `throughput_tuning` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 过大批处理会增大时延和显存压力
- combo_effects: N/A

## Development View

- definition_ref: vllm/engine/arg_utils.py:877
- read_ref: vllm/vllm/config/parallel.py:169, vllm/vllm/config/parallel.py:368, vllm/vllm/config/parallel.py:372
- effect_ref: vllm/vllm/config/parallel.py:368, vllm/vllm/config/parallel.py:372, vllm/vllm/engine/arg_utils.py:877
- web_refs: 3

## Details/Edge Cases

- failure_modes: TTFT/TPOT 退化; OOM
- value_failure_signals: TTFT/TPOT 退化; OOM
- recommendation: 按 TTFT/TPOT/吞吐三指标联合调参。
- updated_at: 2026-03-11
