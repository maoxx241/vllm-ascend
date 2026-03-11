---
topic_id: vllm.arg.async_scheduling
canonical_term: --async-scheduling
topic_kind: parameter
---

# --async-scheduling

## Core

- topic_id: `vllm.arg.async_scheduling`
- canonical_term: `--async-scheduling`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `throughput_tuning`
- status/confidence: `aligned` / `0.95`
- source: `code` / source_tags: code
- semantics: 调度和批处理参数调优，目标提升吞吐。
- aliases: `--async-scheduling`, `async-scheduling`, `async_scheduling`, `async scheduling`, `asyncscheduling`, `throughput_tuning`, `throughput tuning`, `throughput-tuning`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `throughput_tuning` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时，系统会在兼容场景自动开启，不兼容时自动关闭。
- value_shape: `binary_toggle`
- accepted_values: enabled, disabled
- constraints: 仅支持 distributed_executor_backend in {mp, uni, external_launcher}; 与 disable_padded_drafter_batch=True 不兼容; Mamba prefix cache 模式（mamba_cache_mode != none）下不兼容
- combo_effects: 与部分 speculative decoding 组合会被强制关闭或报错; 启用后会影响 DP 同步策略（默认倾向 disable NCCL for DP sync）

## Development View

- definition_ref: vllm/engine/arg_utils.py:1157
- read_ref: vllm/vllm/config/scheduler.py:131, vllm/vllm/config/scheduler.py:155, vllm/vllm/config/scheduler.py:203
- effect_ref: vllm/vllm/config/scheduler.py:155, vllm/vllm/config/vllm.py:627, vllm/vllm/config/vllm.py:656
- web_refs: 5

## Details/Edge Cases

- failure_modes: TTFT/TPOT 退化; OOM
- value_failure_signals: ValueError: async scheduling only supports mp/uni/external_launcher; ValueError: not compatible with disable_padded_drafter_batch=True
- recommendation: 按 TTFT/TPOT/吞吐三指标联合调参。
- updated_at: 2026-03-11
