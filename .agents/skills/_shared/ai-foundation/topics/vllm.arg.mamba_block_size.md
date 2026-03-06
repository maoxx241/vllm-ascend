---
topic_id: vllm.arg.mamba_block_size
canonical_term: --mamba-block-size
topic_kind: parameter
---

# --mamba-block-size

## Core

- topic_id: `vllm.arg.mamba_block_size`
- canonical_term: `--mamba-block-size`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `memory_tuning`
- status/confidence: `aligned` / `0.88`
- source: `code` / source_tags: code
- semantics: 控制 KV/权重/中间缓存占用，平衡容量与性能。
- aliases: `--mamba-block-size`, `mamba-block-size`, `mamba_block_size`, `mamba block size`, `mambablocksize`, `memory_tuning`, `memory tuning`, `memory-tuning`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `memory_tuning` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 过激参数容易触发 OOM
- combo_effects: N/A

## Development View

- definition_ref: vllm/engine/arg_utils.py:959
- read_ref: vllm/vllm/config/cache.py:113, vllm/vllm/config/vllm.py:1479, vllm/vllm/config/vllm.py:1480
- effect_ref: vllm/vllm/model_executor/layers/mamba/mamba_mixer2.py:704, vllm/vllm/model_executor/models/config.py:371, vllm/vllm/model_executor/models/config.py:379
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动或运行 OOM; 缓存不足导致吞吐下降
- value_failure_signals: 启动或运行 OOM; 缓存不足导致吞吐下降
- recommendation: 先保守设置，再渐进放大。
- updated_at: 2026-03-06
