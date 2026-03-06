---
topic_id: vllm.arg.max_cudagraph_capture_size
canonical_term: --max-cudagraph-capture-size
topic_kind: parameter
---

# --max-cudagraph-capture-size

## Core

- topic_id: `vllm.arg.max_cudagraph_capture_size`
- canonical_term: `--max-cudagraph-capture-size`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `graph_mode`
- status/confidence: `needs_manual_review` / `0.79`
- source: `code` / source_tags: code
- semantics: 控制 eager/graph 执行策略，通常优化吞吐与时延抖动。
- aliases: `--max-cudagraph-capture-size`, `max-cudagraph-capture-size`, `max_cudagraph_capture_size`, `max cudagraph capture size`, `maxcudagraphcapturesize`, `graph_mode`, `graph mode`, `graph-mode`

## Foundation

- Ascend 图模式由 ACLGraph 与 Xlite 图配置共同作用，FULL_DECODE_ONLY 常用于先稳态加速 decode。
- 推荐结合 feature: `graph_mode` 查看稳定原理。

## Deployment View

- default_behavior: 默认 None；未设时按 min(max_num_seqs * decode_query_len * 2, 512) 自动估算。
- value_shape: `numeric`
- accepted_values: int >= 1 when cudagraph enabled
- constraints: 与 compilation_config.max_cudagraph_capture_size 互斥。; 若同时显式给出 cudagraph_capture_sizes，需与其最大值一致，否则报错。
- combo_effects: 与 --cudagraph-capture-sizes、--max-num-batched-tokens 联动决定最终可用 sizes。

## Development View

- definition_ref: vllm/engine/arg_utils.py:1168
- read_ref: vllm/vllm/config/compilation.py:334, vllm/vllm/config/compilation.py:335, vllm/vllm/config/compilation.py:571
- effect_ref: vllm/vllm/config/compilation.py:582, vllm/vllm/config/compilation.py:1138, vllm/vllm/config/compilation.py:1142
- web_refs: 5

## Details/Edge Cases

- failure_modes: 图编译失败; 服务启动后首轮请求异常
- value_failure_signals: ValueError: max_cudagraph_capture_size and compilation_config... are mutually exclusive; ValueError: customized max_cudagraph_capture_size ... should be consistent ...
- recommendation: 先小流量验证 FULL_DECODE_ONLY，再放量。
- updated_at: 2026-03-06
