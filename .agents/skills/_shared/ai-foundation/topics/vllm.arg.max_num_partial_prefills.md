---
topic_id: vllm.arg.max_num_partial_prefills
canonical_term: --max-num-partial-prefills
topic_kind: parameter
---

# --max-num-partial-prefills

## Core

- topic_id: `vllm.arg.max_num_partial_prefills`
- canonical_term: `--max-num-partial-prefills`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `prefill_decode_disaggregation`
- status/confidence: `aligned` / `0.88`
- source: `code` / source_tags: code
- semantics: 预填充与解码分离部署，优化资源利用和吞吐扩展。
- aliases: `--max-num-partial-prefills`, `max-num-partial-prefills`, `max_num_partial_prefills`, `max num partial prefills`, `maxnumpartialprefills`, `prefill_decode_disaggregation`, `prefill decode disaggregation`, `prefill-decode-disaggregation`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `prefill_decode_disaggregation` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 单机简化部署无法完整覆盖
- combo_effects: N/A

## Development View

- definition_ref: vllm/engine/arg_utils.py:1119
- read_ref: vllm/vllm/config/scheduler.py:62, vllm/vllm/config/scheduler.py:69, vllm/vllm/config/scheduler.py:229
- effect_ref: vllm/vllm/config/scheduler.py:229, vllm/vllm/config/scheduler.py:273, vllm/vllm/config/scheduler.py:287
- web_refs: 3

## Details/Edge Cases

- failure_modes: connector 超时; P/D 节点路由异常
- value_failure_signals: connector 超时; P/D 节点路由异常
- recommendation: 先验证连接器与地址，再调并行参数。
- updated_at: 2026-03-06
