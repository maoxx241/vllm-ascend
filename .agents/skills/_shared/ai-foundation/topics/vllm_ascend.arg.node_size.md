---
topic_id: vllm_ascend.arg.node_size
canonical_term: --node-size
topic_kind: parameter
---

# --node-size

## Core

- topic_id: `vllm_ascend.arg.node_size`
- canonical_term: `--node-size`
- kind/scope: `arg` / `vllm_ascend`
- stage: `startup`
- primary_feature: `general_runtime`
- status/confidence: `needs_manual_review` / `0.76`
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `--node-size`, `node-size`, `node_size`, `node size`, `nodesize`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: Total number of nodes
- value_shape: `numeric`
- accepted_values: int value
- constraints: 错误组合可能影响稳定性
- combo_effects: N/A

## Development View

- definition_ref: examples/offline_data_parallel.py:83, examples/offline_external_launcher.py:115, examples/offline_weight_load.py:124
- read_ref: vllm-ascend/examples/offline_data_parallel.py:190, vllm-ascend/examples/offline_data_parallel.py:190, vllm-ascend/examples/offline_data_parallel.py:193
- effect_ref: vllm-ascend/examples/offline_data_parallel.py:193, vllm-ascend/examples/offline_external_launcher.py:269, vllm-ascend/examples/offline_weight_load.py:273
- web_refs: 2

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-05
