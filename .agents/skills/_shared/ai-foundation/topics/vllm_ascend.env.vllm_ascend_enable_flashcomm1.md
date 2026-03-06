---
topic_id: vllm_ascend.env.vllm_ascend_enable_flashcomm1
canonical_term: VLLM_ASCEND_ENABLE_FLASHCOMM1
topic_kind: parameter
---

# VLLM_ASCEND_ENABLE_FLASHCOMM1

## Core

- topic_id: `vllm_ascend.env.vllm_ascend_enable_flashcomm1`
- canonical_term: `VLLM_ASCEND_ENABLE_FLASHCOMM1`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `throughput_tuning`
- status/confidence: `aligned` / `0.95`
- source: `multi_source` / source_tags: code_definition, code_reference, docs_export, tests_yaml
- semantics: 开启 FlashComm1 通信优化，常用于高并发场景。
- aliases: `VLLM_ASCEND_ENABLE_FLASHCOMM1`, `vllm_ascend_enable_flashcomm1`, `vllm-ascend-enable-flashcomm1`, `vllm ascend enable flashcomm1`, `throughput_tuning`, `throughput tuning`, `throughput-tuning`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `throughput_tuning` 查看稳定原理。

## Deployment View

- default_behavior: 默认 0（关闭）。
- value_shape: `binary_toggle`
- accepted_values: 0, 1
- constraints: 主要在 MoE 且 tp_size > 1 场景有收益
- combo_effects: 与 prefill_context_parallel_size、tensor_parallel_size 联动时会约束 max_num_batched_tokens 对齐

## Development View

- definition_ref: docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:155, docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:320, docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:88
- read_ref: vllm-ascend/vllm_ascend/envs.py:74, vllm-ascend/vllm_ascend/envs.py:74, vllm-ascend/vllm_ascend/platform.py:406
- effect_ref: vllm-ascend/vllm_ascend/platform.py:406, vllm-ascend/vllm_ascend/utils.py:1001
- web_refs: 4

## Details/Edge Cases

- failure_modes: TTFT/TPOT 退化; OOM
- value_failure_signals: 不满足约束时会触发参数对齐告警或收益不稳定
- recommendation: 按 TTFT/TPOT/吞吐三指标联合调参。
- updated_at: 2026-03-06
