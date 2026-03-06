---
topic_id: vllm_ascend.env.task_queue_enable
canonical_term: TASK_QUEUE_ENABLE
topic_kind: parameter
---

# TASK_QUEUE_ENABLE

## Core

- topic_id: `vllm_ascend.env.task_queue_enable`
- canonical_term: `TASK_QUEUE_ENABLE`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `model_selection`
- status/confidence: `upstream_delta` / `0.55`
- source: `multi_source` / source_tags: docs_export, tests_yaml
- semantics: 控制 Ascend 任务队列执行模式，常与高并发服务配置同时启用。
- aliases: `TASK_QUEUE_ENABLE`, `task_queue_enable`, `task-queue-enable`, `task queue enable`, `model_selection`, `model selection`, `model-selection`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `model_selection` 查看稳定原理。

## Deployment View

- default_behavior: 在多数 Ascend 部署样例中推荐设置为 1。
- value_shape: `binary_toggle`
- accepted_values: 0, 1
- constraints: 需结合模型并发参数与通信参数联合验证。
- combo_effects: 常与 HCCL_OP_EXPANSION_MODE、--max-num-batched-tokens、图模式配置同时调优。

## Development View

- definition_ref: docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md:160, docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:160, docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:226
- read_ref: N/A
- effect_ref: N/A
- web_refs: 5

## Details/Edge Cases

- failure_modes: 加载失败; 返回格式异常
- value_failure_signals: 高并发下吞吐抖动或稳定性下降。
- recommendation: 固定模型版本并记录依赖。
- updated_at: 2026-03-06
