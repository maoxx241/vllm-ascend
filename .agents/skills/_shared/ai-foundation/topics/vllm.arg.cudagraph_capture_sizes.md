---
topic_id: vllm.arg.cudagraph_capture_sizes
canonical_term: --cudagraph-capture-sizes
topic_kind: parameter
---

# --cudagraph-capture-sizes

## Core

- topic_id: `vllm.arg.cudagraph_capture_sizes`
- canonical_term: `--cudagraph-capture-sizes`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `graph_mode`
- status/confidence: `needs_manual_review` / `0.79`
- semantics: 控制 eager/graph 执行策略，通常优化吞吐与时延抖动。
- aliases: `--cudagraph-capture-sizes`, `cudagraph-capture-sizes`, `cudagraph_capture_sizes`, `cudagraph capture sizes`, `cudagraphcapturesizes`, `graph_mode`, `graph mode`, `graph-mode`

## Foundation

- 图模式通过稳定执行图降低调度抖动，提升吞吐稳定性。
- 推荐结合 feature: `graph_mode` 查看稳定原理。

## Deployment View

- default_behavior: 默认 None；未指定时按 max_num_seqs/max_num_batched_tokens 自动生成候选 sizes。
- value_shape: `list_numeric`
- accepted_values: list[int] (non-empty when cudagraph enabled)
- constraints: 与 compilation_config.cudagraph_capture_sizes 互斥。; 开启 cudagraph 时列表不能为空。
- combo_effects: 与 --max-cudagraph-capture-size 需一致，否则可能触发告警或错误。; 与 speculative/sequence parallel 参数联动后，sizes 可能被重新对齐。

## Development View

- definition_ref: vllm/engine/arg_utils.py:1165
- read_ref: vllm/vllm/compilation/piecewise_backend.py:119, vllm/vllm/compilation/piecewise_backend.py:121, vllm/vllm/config/compilation.py:332
- effect_ref: vllm/vllm/config/compilation.py:574, vllm/vllm/config/compilation.py:935, vllm/vllm/config/compilation.py:1130
- web_refs: 5

## Details/Edge Cases

- failure_modes: 图编译失败; 服务启动后首轮请求异常
- value_failure_signals: ValueError: cudagraph_capture_sizes and compilation_config... are mutually exclusive; AssertionError: cudagraph_capture_sizes should contain at least one element
- recommendation: 先小流量验证 FULL_DECODE_ONLY，再放量。
- updated_at: 2026-03-05
