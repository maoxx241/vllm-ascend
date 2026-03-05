---
topic_id: vllm_ascend.env.triton_all_blocks_parallel
canonical_term: TRITON_ALL_BLOCKS_PARALLEL
topic_kind: parameter
---

# TRITON_ALL_BLOCKS_PARALLEL

## Core

- topic_id: `vllm_ascend.env.triton_all_blocks_parallel`
- canonical_term: `TRITON_ALL_BLOCKS_PARALLEL`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `upstream_delta` / `0.68`
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `TRITON_ALL_BLOCKS_PARALLEL`, `triton_all_blocks_parallel`, `triton-all-blocks-parallel`, `triton all blocks parallel`, `general_runtime`, `general runtime`, `general-runtime`

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

- definition_ref: vllm_ascend/ops/rotary_embedding.py:497
- read_ref: vllm-ascend/vllm_ascend/ops/rotary_embedding.py:494, vllm-ascend/vllm_ascend/ops/rotary_embedding.py:496, vllm-ascend/vllm_ascend/ops/rotary_embedding.py:497
- effect_ref: vllm-ascend/vllm_ascend/ops/rotary_embedding.py:494, vllm-ascend/vllm_ascend/ops/rotary_embedding.py:496
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-05
