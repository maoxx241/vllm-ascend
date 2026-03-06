---
topic_id: vllm.arg.max_model_len
canonical_term: --max-model-len
topic_kind: parameter
---

# --max-model-len

## Core

- topic_id: `vllm.arg.max_model_len`
- canonical_term: `--max-model-len`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `memory_tuning`
- status/confidence: `aligned` / `0.95`
- source: `code` / source_tags: code
- semantics: 控制 KV/权重/中间缓存占用，平衡容量与性能。
- aliases: `--max-model-len`, `max-model-len`, `max_model_len`, `max model len`, `maxmodellen`, `memory_tuning`, `memory tuning`, `memory-tuning`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `memory_tuning` 查看稳定原理。

## Deployment View

- default_behavior: 未设置时按模型配置推导；-1/auto 触发自动适配。
- value_shape: `numeric_or_auto`
- accepted_values: int >= 1, -1, auto, k/m/g 后缀
- constraints: 超过模型推导上限需设置 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 才允许; 过大在 RoPE/绝对位置编码模型上可能导致 NaN/OOB 风险
- combo_effects: 与 max_num_batched_tokens、sliding_window、speculative 配置联动

## Development View

- definition_ref: vllm/engine/arg_utils.py:669
- read_ref: vllm/vllm/benchmarks/latency.py:89, vllm/vllm/benchmarks/latency.py:92, vllm/vllm/benchmarks/mm_processor.py:158
- effect_ref: vllm/vllm/config/model.py:657, vllm/vllm/config/model.py:1550, vllm/vllm/config/model.py:1938
- web_refs: 6

## Details/Edge Cases

- failure_modes: 启动或运行 OOM; 缓存不足导致吞吐下降
- value_failure_signals: ValueError: user-specified max_model_len greater than derived limit; 超长位置导致 NaN 或越界
- recommendation: 先保守设置，再渐进放大。
- updated_at: 2026-03-06
