---
topic_id: vllm_ascend.arg.proc_per_node
canonical_term: --proc-per-node
topic_kind: parameter
---

# --proc-per-node

## Core

- topic_id: `vllm_ascend.arg.proc_per_node`
- canonical_term: `--proc-per-node`
- kind/scope: `arg` / `vllm_ascend`
- stage: `startup`
- primary_feature: `general_runtime`
- status/confidence: `needs_manual_review` / `0.76`
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `--proc-per-node`, `proc-per-node`, `proc_per_node`, `proc per node`, `procpernode`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: Number of processes per node
- value_shape: `numeric`
- accepted_values: int value
- constraints: 错误组合可能影响稳定性
- combo_effects: N/A

## Development View

- definition_ref: examples/offline_external_launcher.py:117, examples/offline_weight_load.py:126
- read_ref: vllm-ascend/examples/offline_external_launcher.py:266, vllm-ascend/examples/offline_external_launcher.py:266, vllm-ascend/examples/offline_external_launcher.py:276
- effect_ref: vllm-ascend/examples/offline_external_launcher.py:266, vllm-ascend/examples/offline_external_launcher.py:266, vllm-ascend/examples/offline_external_launcher.py:276
- web_refs: 2

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-05
