---
topic_id: vllm.env.vllm_enable_cudagraph_gc
canonical_term: VLLM_ENABLE_CUDAGRAPH_GC
topic_kind: parameter
---

# VLLM_ENABLE_CUDAGRAPH_GC

## Core

- topic_id: `vllm.env.vllm_enable_cudagraph_gc`
- canonical_term: `VLLM_ENABLE_CUDAGRAPH_GC`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `graph_mode`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: 控制 eager/graph 执行策略，通常优化吞吐与时延抖动。
- aliases: `VLLM_ENABLE_CUDAGRAPH_GC`, `vllm_enable_cudagraph_gc`, `vllm-enable-cudagraph-gc`, `vllm enable cudagraph gc`, `graph_mode`, `graph mode`, `graph-mode`

## Foundation

- Ascend 图模式由 ACLGraph 与 Xlite 图配置共同作用，FULL_DECODE_ONLY 常用于先稳态加速 decode。
- 推荐结合 feature: `graph_mode` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `numeric`
- accepted_values: int value
- constraints: 部分动态路径可能要求 eager 回退
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:1370
- read_ref: vllm/vllm/envs.py:193, vllm/vllm/envs.py:1370, vllm/vllm/envs.py:1371
- effect_ref: vllm/vllm/envs.py:193, vllm/vllm/envs.py:1370, vllm/vllm/envs.py:1371
- web_refs: 4

## Details/Edge Cases

- failure_modes: 图编译失败; 服务启动后首轮请求异常
- value_failure_signals: 图编译失败; 服务启动后首轮请求异常
- recommendation: 先小流量验证 FULL_DECODE_ONLY，再放量。
- updated_at: 2026-03-06
