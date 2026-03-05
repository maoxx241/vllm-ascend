---
topic_id: vllm.arg.enable_lora
canonical_term: --enable-lora
topic_kind: parameter
---

# --enable-lora

## Core

- topic_id: `vllm.arg.enable_lora`
- canonical_term: `--enable-lora`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `lora`
- status/confidence: `needs_manual_review` / `0.86`
- semantics: 开启 LoRA 适配器加载与路由。
- aliases: `--enable-lora`, `enable-lora`, `enable_lora`, `enable lora`, `enablelora`, `lora`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `lora` 查看稳定原理。

## Deployment View

- default_behavior: If True, enable handling of LoRA adapters.
- value_shape: `binary_or_auto`
- accepted_values: enabled, disabled, unset(auto)
- constraints: 工件缺失或不匹配会导致加载失败
- combo_effects: N/A

## Development View

- definition_ref: vllm/engine/arg_utils.py:1027
- read_ref: vllm/vllm/benchmarks/throughput.py:89, vllm/vllm/benchmarks/throughput.py:638, vllm/vllm/benchmarks/throughput.py:640
- effect_ref: vllm/vllm/benchmarks/throughput.py:89, vllm/vllm/benchmarks/throughput.py:638, vllm/vllm/benchmarks/throughput.py:640
- web_refs: 6

## Details/Edge Cases

- failure_modes: LoRA 模块加载报错; 输出异常
- value_failure_signals: LoRA 模块加载报错; 输出异常
- recommendation: 先离线验证 LoRA 工件，再接入在线服务。
- updated_at: 2026-03-05
