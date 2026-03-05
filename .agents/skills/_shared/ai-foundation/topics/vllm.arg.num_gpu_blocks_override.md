---
topic_id: vllm.arg.num_gpu_blocks_override
canonical_term: --num-gpu-blocks-override
topic_kind: parameter
---

# --num-gpu-blocks-override

## Core

- topic_id: `vllm.arg.num_gpu_blocks_override`
- canonical_term: `--num-gpu-blocks-override`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `general_runtime`
- status/confidence: `needs_manual_review` / `0.76`
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `--num-gpu-blocks-override`, `num-gpu-blocks-override`, `num_gpu_blocks_override`, `num gpu blocks override`, `numgpublocksoverride`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 错误组合可能影响稳定性
- combo_effects: N/A

## Development View

- definition_ref: vllm/engine/arg_utils.py:933
- read_ref: vllm/vllm/config/cache.py:70, vllm/vllm/config/cache.py:188, vllm/vllm/engine/arg_utils.py:493
- effect_ref: vllm/vllm/v1/core/kv_cache_utils.py:815, vllm/vllm/v1/core/kv_cache_utils.py:817
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-05
