---
topic_id: vllm_ascend.arg.temperature
canonical_term: --temperature
topic_kind: parameter
---

# --temperature

## Core

- topic_id: `vllm_ascend.arg.temperature`
- canonical_term: `--temperature`
- kind/scope: `arg` / `vllm_ascend`
- stage: `startup`
- primary_feature: `general_runtime`
- status/confidence: `needs_manual_review` / `0.76`
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `--temperature`, `temperature`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: Float that controls the randomness of the sampling.
- value_shape: `numeric`
- accepted_values: float value
- constraints: 错误组合可能影响稳定性
- combo_effects: N/A

## Development View

- definition_ref: examples/offline_external_launcher.py:126, examples/offline_weight_load.py:135
- read_ref: vllm/vllm/benchmarks/latency.py:98, vllm/vllm/benchmarks/mm_processor.py:172, vllm/vllm/benchmarks/serve.py:1418
- effect_ref: vllm/vllm/benchmarks/serve.py:1635, vllm/vllm/config/model.py:267, vllm/vllm/entrypoints/grpc_server.py:287
- web_refs: 2

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-05
