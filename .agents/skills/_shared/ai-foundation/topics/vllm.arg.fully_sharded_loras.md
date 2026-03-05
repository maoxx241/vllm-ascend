---
topic_id: vllm.arg.fully_sharded_loras
canonical_term: --fully-sharded-loras
topic_kind: parameter
---

# --fully-sharded-loras

## Core

- topic_id: `vllm.arg.fully_sharded_loras`
- canonical_term: `--fully-sharded-loras`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `lora`
- status/confidence: `needs_manual_review` / `0.79`
- semantics: 开启 LoRA 适配器加载与路由。
- aliases: `--fully-sharded-loras`, `fully-sharded-loras`, `fully_sharded_loras`, `fully sharded loras`, `fullyshardedloras`, `lora`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `lora` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 工件缺失或不匹配会导致加载失败
- combo_effects: N/A

## Development View

- definition_ref: vllm/engine/arg_utils.py:1043
- read_ref: vllm/vllm/config/lora.py:36, vllm/vllm/config/lora.py:84, vllm/vllm/engine/arg_utils.py:486
- effect_ref: vllm/vllm/lora/layers/base_linear.py:50, vllm/vllm/lora/layers/base_linear.py:59, vllm/vllm/lora/layers/column_parallel_linear.py:210
- web_refs: 5

## Details/Edge Cases

- failure_modes: LoRA 模块加载报错; 输出异常
- value_failure_signals: LoRA 模块加载报错; 输出异常
- recommendation: 先离线验证 LoRA 工件，再接入在线服务。
- updated_at: 2026-03-05
