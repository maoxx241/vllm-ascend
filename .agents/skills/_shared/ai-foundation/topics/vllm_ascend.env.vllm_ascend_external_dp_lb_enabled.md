---
topic_id: vllm_ascend.env.vllm_ascend_external_dp_lb_enabled
canonical_term: VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED
topic_kind: parameter
---

# VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED

## Core

- topic_id: `vllm_ascend.env.vllm_ascend_external_dp_lb_enabled`
- canonical_term: `VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `data_parallel`
- status/confidence: `upstream_delta` / `0.58`
- source: `docs_export` / source_tags: docs_export
- semantics: 通过副本扩展吞吐能力，并依赖 DP 地址和 RPC 协调。
- aliases: `VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED`, `vllm_ascend_external_dp_lb_enabled`, `vllm-ascend-external-dp-lb-enabled`, `vllm ascend external dp lb enabled`, `data_parallel`, `data parallel`, `data-parallel`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `data_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 错误地址/端口会导致调度与健康检查失败
- combo_effects: N/A

## Development View

- definition_ref: docs/source/user_guide/feature_guide/large_scale_ep.md:141, docs/source/user_guide/feature_guide/large_scale_ep.md:208
- read_ref: N/A
- effect_ref: N/A
- web_refs: 5

## Details/Edge Cases

- failure_modes: RPC 连接失败; 请求分发不均衡
- value_failure_signals: RPC 连接失败; 请求分发不均衡
- recommendation: 固定 DP 地址和端口后再迭代性能参数。
- updated_at: 2026-03-11
