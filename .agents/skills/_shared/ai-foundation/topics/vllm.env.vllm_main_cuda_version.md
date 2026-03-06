---
topic_id: vllm.env.vllm_main_cuda_version
canonical_term: VLLM_MAIN_CUDA_VERSION
topic_kind: parameter
---

# VLLM_MAIN_CUDA_VERSION

## Core

- topic_id: `vllm.env.vllm_main_cuda_version`
- canonical_term: `VLLM_MAIN_CUDA_VERSION`
- kind/scope: `env` / `vllm`
- stage: `build`
- primary_feature: `general_runtime`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `VLLM_MAIN_CUDA_VERSION`, `vllm_main_cuda_version`, `vllm-main-cuda-version`, `vllm main cuda version`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 错误组合可能影响稳定性
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:465
- read_ref: vllm/vllm/envs.py:75, vllm/vllm/envs.py:465, vllm/vllm/envs.py:465
- effect_ref: vllm/vllm/envs.py:75, vllm/vllm/envs.py:465, vllm/vllm/envs.py:465
- web_refs: 2

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-06
