---
topic_id: vllm_ascend.env.hccl_socket_ifname
canonical_term: HCCL_SOCKET_IFNAME
topic_kind: parameter
---

# HCCL_SOCKET_IFNAME

## Core

- topic_id: `vllm_ascend.env.hccl_socket_ifname`
- canonical_term: `HCCL_SOCKET_IFNAME`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `general_runtime`
- status/confidence: `upstream_delta` / `0.55`
- source: `docs_export` / source_tags: docs_export
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `HCCL_SOCKET_IFNAME`, `hccl_socket_ifname`, `hccl-socket-ifname`, `hccl socket ifname`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: HCCL_SOCKET_IFNAME 未设置时使用 HCCL 默认行为。
- value_shape: `runtime_string_or_numeric`
- accepted_values: 由 HCCL 文档定义，常见为整数/枚举字符串
- constraints: 不同 CANN/HCCL 版本支持范围不同，应与平台版本矩阵对齐。
- combo_effects: 与 TP/DP/CP 等并行参数耦合，配置不当会导致通信性能下降或初始化失败。

## Development View

- definition_ref: docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:152, docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:219, docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:85
- read_ref: N/A
- effect_ref: N/A
- web_refs: 5

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: HCCL init failed; Communication timeout
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-06
