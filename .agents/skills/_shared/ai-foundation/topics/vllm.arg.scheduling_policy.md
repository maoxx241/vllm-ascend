---
topic_id: vllm.arg.scheduling_policy
canonical_term: --scheduling-policy
topic_kind: parameter
---

# --scheduling-policy

## Core

- topic_id: `vllm.arg.scheduling_policy`
- canonical_term: `--scheduling-policy`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `general_runtime`
- status/confidence: `aligned` / `0.88`
- source: `code` / source_tags: code
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `--scheduling-policy`, `scheduling-policy`, `scheduling_policy`, `scheduling policy`, `schedulingpolicy`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: 默认 fcfs。
- value_shape: `enum`
- accepted_values: fcfs, priority
- constraints: 仅在 v1 scheduler 语义下生效
- combo_effects: 与请求优先级字段联合生效，不设置优先级时接近 FCFS 行为

## Development View

- definition_ref: vllm/engine/arg_utils.py:1137
- read_ref: vllm/vllm/engine/arg_utils.py:534, vllm/vllm/engine/arg_utils.py:1688, vllm/vllm/engine/arg_utils.py:1138
- effect_ref: vllm/vllm/engine/arg_utils.py:534, vllm/vllm/engine/arg_utils.py:1688, vllm/vllm/engine/arg_utils.py:1138
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 非法值会在参数校验时报错
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-11
