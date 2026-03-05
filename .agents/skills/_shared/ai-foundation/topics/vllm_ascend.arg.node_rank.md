---
topic_id: vllm_ascend.arg.node_rank
canonical_term: --node-rank
topic_kind: parameter
---

# --node-rank

## Core

- topic_id: `vllm_ascend.arg.node_rank`
- canonical_term: `--node-rank`
- kind/scope: `arg` / `vllm_ascend`
- stage: `startup`
- primary_feature: `general_runtime`
- status/confidence: `needs_manual_review` / `0.83`
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `--node-rank`, `node-rank`, `node_rank`, `node rank`, `noderank`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: Rank of the current node
- value_shape: `numeric`
- accepted_values: int value
- constraints: 错误组合可能影响稳定性
- combo_effects: N/A

## Development View

- definition_ref: examples/offline_data_parallel.py:84, examples/offline_external_launcher.py:116, examples/offline_weight_load.py:125
- read_ref: vllm/vllm/config/parallel.py:230, vllm/vllm/config/parallel.py:464, vllm/vllm/config/parallel.py:527
- effect_ref: vllm/vllm/config/parallel.py:464, vllm/vllm/engine/arg_utils.py:1529
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-05
