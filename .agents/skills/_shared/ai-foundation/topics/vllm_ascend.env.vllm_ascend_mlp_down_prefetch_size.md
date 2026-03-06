---
topic_id: vllm_ascend.env.vllm_ascend_mlp_down_prefetch_size
canonical_term: VLLM_ASCEND_MLP_DOWN_PREFETCH_SIZE
topic_kind: parameter
---

# VLLM_ASCEND_MLP_DOWN_PREFETCH_SIZE

## Core

- topic_id: `vllm_ascend.env.vllm_ascend_mlp_down_prefetch_size`
- canonical_term: `VLLM_ASCEND_MLP_DOWN_PREFETCH_SIZE`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `weight_prefetch`
- status/confidence: `aligned` / `0.98`
- source: `code` / source_tags: code_definition, code_reference
- semantics: 提前预取权重块，降低 decode 等待。
- aliases: `VLLM_ASCEND_MLP_DOWN_PREFETCH_SIZE`, `vllm_ascend_mlp_down_prefetch_size`, `vllm-ascend-mlp-down-prefetch-size`, `vllm ascend mlp down prefetch size`, `weight_prefetch`, `weight prefetch`, `weight-prefetch`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `weight_prefetch` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `numeric`
- accepted_values: int value
- constraints: 显存紧张场景可能增加压力
- combo_effects: N/A

## Development View

- definition_ref: vllm_ascend/ascend_config.py:153, vllm_ascend/envs.py:87, vllm_ascend/envs.py:88
- read_ref: vllm-ascend/vllm_ascend/ascend_config.py:153, vllm-ascend/vllm_ascend/envs.py:87, vllm-ascend/vllm_ascend/envs.py:88
- effect_ref: vllm-ascend/vllm_ascend/ascend_config.py:153, vllm-ascend/vllm_ascend/envs.py:87, vllm-ascend/vllm_ascend/envs.py:88
- web_refs: 5

## Details/Edge Cases

- failure_modes: 显存 OOM; 收益不稳定
- value_failure_signals: 显存 OOM; 收益不稳定
- recommendation: 与 max_model_len/gpu_memory_utilization 联动调优。
- updated_at: 2026-03-06
