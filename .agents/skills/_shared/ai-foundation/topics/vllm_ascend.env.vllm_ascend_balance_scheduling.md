---
topic_id: vllm_ascend.env.vllm_ascend_balance_scheduling
canonical_term: VLLM_ASCEND_BALANCE_SCHEDULING
topic_kind: parameter
---

# VLLM_ASCEND_BALANCE_SCHEDULING

## Core

- topic_id: `vllm_ascend.env.vllm_ascend_balance_scheduling`
- canonical_term: `VLLM_ASCEND_BALANCE_SCHEDULING`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `throughput_tuning`
- status/confidence: `aligned` / `0.95`
- source: `multi_source` / source_tags: code_definition, code_reference, docs_export
- semantics: 调度和批处理参数调优，目标提升吞吐。
- aliases: `VLLM_ASCEND_BALANCE_SCHEDULING`, `vllm_ascend_balance_scheduling`, `vllm-ascend-balance-scheduling`, `vllm ascend balance scheduling`, `throughput_tuning`, `throughput tuning`, `throughput-tuning`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `throughput_tuning` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `numeric`
- accepted_values: int value
- constraints: 过大批处理会增大时延和显存压力
- combo_effects: N/A

## Development View

- definition_ref: docs/source/tutorials/models/DeepSeek-R1.md:152, docs/source/tutorials/models/DeepSeek-R1.md:198, docs/source/tutorials/models/DeepSeek-R1.md:97
- read_ref: vllm-ascend/vllm_ascend/envs.py:115, vllm-ascend/vllm_ascend/envs.py:115, vllm-ascend/vllm_ascend/patch/__init__.py:91
- effect_ref: vllm-ascend/vllm_ascend/envs.py:115, vllm-ascend/vllm_ascend/envs.py:115, vllm-ascend/vllm_ascend/patch/__init__.py:91
- web_refs: 4

## Details/Edge Cases

- failure_modes: TTFT/TPOT 退化; OOM
- value_failure_signals: TTFT/TPOT 退化; OOM
- recommendation: 按 TTFT/TPOT/吞吐三指标联合调参。
- updated_at: 2026-03-11
