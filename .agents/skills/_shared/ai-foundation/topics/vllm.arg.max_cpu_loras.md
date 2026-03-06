---
topic_id: vllm.arg.max_cpu_loras
canonical_term: --max-cpu-loras
topic_kind: parameter
---

# --max-cpu-loras

## Core

- topic_id: `vllm.arg.max_cpu_loras`
- canonical_term: `--max-cpu-loras`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `lora`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code
- semantics: 开启 LoRA 适配器加载与路由。
- aliases: `--max-cpu-loras`, `max-cpu-loras`, `max_cpu_loras`, `max cpu loras`, `maxcpuloras`, `lora`

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

- definition_ref: vllm/engine/arg_utils.py:1042
- read_ref: vllm/vllm/config/lora.py:41, vllm/vllm/config/lora.py:93, vllm/vllm/config/lora.py:94
- effect_ref: vllm/vllm/config/lora.py:93, vllm/vllm/config/lora.py:95, vllm/vllm/engine/arg_utils.py:1698
- web_refs: 5

## Details/Edge Cases

- failure_modes: LoRA 模块加载报错; 输出异常
- value_failure_signals: LoRA 模块加载报错; 输出异常
- recommendation: 先离线验证 LoRA 工件，再接入在线服务。
- updated_at: 2026-03-06
