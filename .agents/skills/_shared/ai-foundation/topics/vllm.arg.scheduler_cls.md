---
topic_id: vllm.arg.scheduler_cls
canonical_term: --scheduler-cls
topic_kind: parameter
---

# --scheduler-cls

## Core

- topic_id: `vllm.arg.scheduler_cls`
- canonical_term: `--scheduler-cls`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `throughput_tuning`
- status/confidence: `aligned` / `0.88`
- source: `code` / source_tags: code
- semantics: 调度和批处理参数调优，目标提升吞吐。
- aliases: `--scheduler-cls`, `scheduler-cls`, `scheduler_cls`, `scheduler cls`, `schedulercls`, `throughput_tuning`, `throughput tuning`, `throughput-tuning`

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

- definition_ref: vllm/engine/arg_utils.py:1145
- read_ref: vllm/vllm/config/scheduler.py:118, vllm/vllm/config/scheduler.py:154, vllm/vllm/config/scheduler.py:169
- effect_ref: vllm/vllm/config/scheduler.py:154, vllm/vllm/config/scheduler.py:171, vllm/vllm/config/scheduler.py:172
- web_refs: 3

## Details/Edge Cases

- failure_modes: TTFT/TPOT 退化; OOM
- value_failure_signals: TTFT/TPOT 退化; OOM
- recommendation: 按 TTFT/TPOT/吞吐三指标联合调参。
- updated_at: 2026-03-06
