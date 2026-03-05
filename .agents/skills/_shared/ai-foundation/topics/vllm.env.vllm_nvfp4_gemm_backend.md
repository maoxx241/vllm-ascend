---
topic_id: vllm.env.vllm_nvfp4_gemm_backend
canonical_term: VLLM_NVFP4_GEMM_BACKEND
topic_kind: parameter
---

# VLLM_NVFP4_GEMM_BACKEND

## Core

- topic_id: `vllm.env.vllm_nvfp4_gemm_backend`
- canonical_term: `VLLM_NVFP4_GEMM_BACKEND`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `multimodal`
- status/confidence: `aligned` / `0.91`
- semantics: 控制多模态输入处理和缓存策略。
- aliases: `VLLM_NVFP4_GEMM_BACKEND`, `vllm_nvfp4_gemm_backend`, `vllm-nvfp4-gemm-backend`, `vllm nvfp4 gemm backend`, `multimodal`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `multimodal` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 不支持多模态的模型无法启用相关参数
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:1356
- read_ref: vllm/vllm/envs.py:197, vllm/vllm/envs.py:1356, vllm/vllm/envs.py:1357
- effect_ref: vllm/vllm/model_executor/layers/quantization/utils/nvfp4_utils.py:57
- web_refs: 2

## Details/Edge Cases

- failure_modes: 输入解析失败; 处理时延过高
- value_failure_signals: 输入解析失败; 处理时延过高
- recommendation: 先限制每请求多模态资源，再放开。
- updated_at: 2026-03-05
