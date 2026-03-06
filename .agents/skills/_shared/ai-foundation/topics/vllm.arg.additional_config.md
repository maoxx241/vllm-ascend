---
topic_id: vllm.arg.additional_config
canonical_term: --additional-config
topic_kind: parameter
---

# --additional-config

## Core

- topic_id: `vllm.arg.additional_config`
- canonical_term: `--additional-config`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `general_runtime`
- status/confidence: `needs_manual_review` / `0.83`
- source: `code` / source_tags: code
- semantics: 承载扩展配置，如 weight_prefetch_config 等。
- aliases: `--additional-config`, `additional-config`, `additional_config`, `additional config`, `additionalconfig`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: 默认空字典，不启用额外插件特性。
- value_shape: `json_object`
- accepted_values: xlite_graph_config, weight_prefetch_config, finegrained_tp_config, ascend_compilation_config, eplb_config, refresh, dump_config_path, enable_async_exponential, enable_shared_expert_dp, multistream_overlap_shared_expert, multistream_overlap_gate, recompute_scheduler_enable
- constraints: xlite_graph_config.enabled=true 时要求 block_size=128 且不兼容 speculative decoding; xlite_graph_config.full_mode=true 时应与 FULL/FULL_DECODE_ONLY 图模式联合验证; weight_prefetch_config 需结合并发与模型类型调优 prefetch_ratio; enforce-eager=true 时图相关 additional_config 子字段会退化为无效配置
- combo_effects: 与 --compilation-config、--block-size、并行参数共同决定最终执行路径; 与部分环境变量存在兼容层（如 VLLM_ASCEND_ENABLE_PREFETCH_MLP）; ascend_compilation_config.enable_npugraph_ex 与 ACLGraph 图模式需协同验证

## Development View

- definition_ref: vllm/engine/arg_utils.py:1211
- read_ref: vllm/vllm/config/vllm.py:263, vllm/vllm/config/vllm.py:360, vllm/vllm/config/vllm.py:361
- effect_ref: vllm/vllm/config/vllm.py:360, vllm/vllm/config/vllm.py:361, vllm/vllm/config/vllm.py:361
- web_refs: 5

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: RuntimeError: Xlite graph mode incompatible with current setup; ValueError/AssertionError: finegrained_tp_config 或 eplb_config 校验失败
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-06
