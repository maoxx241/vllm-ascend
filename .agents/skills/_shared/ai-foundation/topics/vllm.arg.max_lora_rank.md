---
topic_id: vllm.arg.max_lora_rank
canonical_term: --max-lora-rank
topic_kind: parameter
---

# --max-lora-rank

## Core

- topic_id: `vllm.arg.max_lora_rank`
- canonical_term: `--max-lora-rank`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `lora`
- status/confidence: `needs_manual_review` / `0.79`
- semantics: 开启 LoRA 适配器加载与路由。
- aliases: `--max-lora-rank`, `max-lora-rank`, `max_lora_rank`, `max lora rank`, `maxlorarank`, `lora`

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

- definition_ref: vllm/engine/arg_utils.py:1033
- read_ref: vllm/vllm/config/lora.py:32, vllm/vllm/config/lora.py:82, vllm/vllm/engine/arg_utils.py:484
- effect_ref: vllm/vllm/lora/layers/base_linear.py:51, vllm/vllm/lora/layers/column_parallel_linear.py:211, vllm/vllm/lora/layers/fused_moe.py:357
- web_refs: 5

## Details/Edge Cases

- failure_modes: LoRA 模块加载报错; 输出异常
- value_failure_signals: LoRA 模块加载报错; 输出异常
- recommendation: 先离线验证 LoRA 工件，再接入在线服务。
- updated_at: 2026-03-05
