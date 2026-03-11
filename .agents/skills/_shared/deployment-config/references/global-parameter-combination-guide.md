---
knowledge_id: deployment-config.global-parameter-combination-guide
domain: deployment-config
knowledge_type: procedure
summary: Combination constraints and profile-level blocks with evidence refs.
last_verified: "2026-03-11"
source_commit: "workspace-head"
freshness: "fresh"
---

# Global Parameter Combination Guide

## Rule Levels

- `hard_block`: must not execute automatically
- `warning`: allow execution with explicit warning and fallback
- `recommended`: preferred baseline for demo deployment

## Rules

1. `recommended.quant_graph_tp` (recommended)
- profile: `*`
- conditions: `quantization, graph_mode, tensor_parallel`
- reason: 量化+图模式+TP 是常见高吞吐组合。
- fallback: `先只启用 quantization+TP，稳定后再加 graph_mode`
2. `warning.weight_prefetch_memory` (warning)
- profile: `*`
- conditions: `weight_prefetch, memory_tuning`
- reason: 权重预取提升吞吐但会增加内存压力。
- fallback: `降低 max_model_len 或 gpu_memory_utilization`
3. `warning.pd_requires_connector` (warning)
- profile: `*`
- conditions: `prefill_decode_disaggregation`
- reason: PD 分离依赖连接器、地址和节点角色配置。
- fallback: `先用单机模板验证，再切换到分离架构`
4. `hard_block.qwen3_32b_w8a8_int4` (hard_block)
- profile: `qwen3-32b-w8a8`
- conditions: `int4_quantization`
- reason: qwen3-32b-w8a8 profile 不提供已验证 int4 工件与路径。
- fallback: `保持 W8A8，或切换到可用 int4 profile`
5. `hard_block.qwen3_32b_w8a8_ep` (hard_block)
- profile: `qwen3-32b-w8a8`
- conditions: `expert_parallel`
- reason: qwen3-32b-w8a8 是 Dense 模型，不适用 EP。
- fallback: `改用 TP/DP 组合调优`

Back to [INDEX](../../INDEX.md).
