---
topic_id: vllm.env.vllm_lora_disable_pdl
canonical_term: VLLM_LORA_DISABLE_PDL
topic_kind: parameter
---

# VLLM_LORA_DISABLE_PDL

## Core

- topic_id: `vllm.env.vllm_lora_disable_pdl`
- canonical_term: `VLLM_LORA_DISABLE_PDL`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `lora`
- status/confidence: `aligned` / `0.91`
- semantics: 开启 LoRA 适配器加载与路由。
- aliases: `VLLM_LORA_DISABLE_PDL`, `vllm_lora_disable_pdl`, `vllm-lora-disable-pdl`, `vllm lora disable pdl`, `lora`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `lora` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `numeric`
- accepted_values: int value
- constraints: 工件缺失或不匹配会导致加载失败
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:1539
- read_ref: vllm/vllm/envs.py:233, vllm/vllm/envs.py:1539, vllm/vllm/envs.py:1539
- effect_ref: vllm/vllm/envs.py:233, vllm/vllm/envs.py:1539, vllm/vllm/envs.py:1539
- web_refs: 4

## Details/Edge Cases

- failure_modes: LoRA 模块加载报错; 输出异常
- value_failure_signals: LoRA 模块加载报错; 输出异常
- recommendation: 先离线验证 LoRA 工件，再接入在线服务。
- updated_at: 2026-03-05
