---
topic_id: vllm.arg.max_loras
canonical_term: --max-loras
topic_kind: parameter
---

# --max-loras

## Core

- topic_id: `vllm.arg.max_loras`
- canonical_term: `--max-loras`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `lora`
- status/confidence: `needs_manual_review` / `0.79`
- semantics: 开启 LoRA 适配器加载与路由。
- aliases: `--max-loras`, `max-loras`, `max_loras`, `max loras`, `maxloras`, `lora`

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

- definition_ref: vllm/engine/arg_utils.py:1032
- read_ref: vllm/vllm/_custom_ops.py:2109, vllm/vllm/_custom_ops.py:2124, vllm/vllm/benchmarks/datasets.py:156
- effect_ref: vllm/vllm/benchmarks/datasets.py:175, vllm/vllm/benchmarks/mm_processor.py:117, vllm/vllm/config/lora.py:95
- web_refs: 5

## Details/Edge Cases

- failure_modes: LoRA 模块加载报错; 输出异常
- value_failure_signals: LoRA 模块加载报错; 输出异常
- recommendation: 先离线验证 LoRA 工件，再接入在线服务。
- updated_at: 2026-03-05
