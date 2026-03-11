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
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code
- semantics: 控制 eager/graph 执行策略，通常优化吞吐与时延抖动。
- aliases: `--cudagraph-capture-sizes`, `cudagraph-capture-sizes`, `cudagraph_capture_sizes`, `cudagraph capture sizes`, `cudagraphcapturesizes`, `graph_mode`, `graph mode`, `graph-mode`

## Foundation

- Ascend 图模式由 ACLGraph 与 Xlite 图配置共同作用，FULL_DECODE_ONLY 常用于先稳态加速 decode。
- 推荐结合 feature: `graph_mode` 查看稳定原理。

## Deployment View

- default_behavior: 默认 None；未指定时按 max_num_seqs/max_num_batched_tokens 自动生成候选 sizes。
- value_shape: `list_numeric`
- accepted_values: list[int] (non-empty when cudagraph enabled)
- constraints: 与 compilation_config.cudagraph_capture_sizes 互斥。; 开启 cudagraph 时列表不能为空。
- combo_effects: 与 --max-cudagraph-capture-size 需一致，否则可能触发告警或错误。; 与 speculative/sequence parallel 参数联动后，sizes 可能被重新对齐。

## Development View

- definition_ref: vllm/engine/arg_utils.py:1170
- read_ref: vllm/vllm/compilation/piecewise_backend.py:119, vllm/vllm/compilation/piecewise_backend.py:121, vllm/vllm/config/compilation.py:331
- effect_ref: vllm/vllm/config/compilation.py:573, vllm/vllm/config/compilation.py:934, vllm/vllm/config/compilation.py:1129
- web_refs: 5

## Details/Edge Cases

- failure_modes: 图编译失败; 服务启动后首轮请求异常
- value_failure_signals: ValueError: cudagraph_capture_sizes and compilation_config... are mutually exclusive; AssertionError: cudagraph_capture_sizes should contain at least one element
- recommendation: 先小流量验证 FULL_DECODE_ONLY，再放量。
- updated_at: 2026-03-11
