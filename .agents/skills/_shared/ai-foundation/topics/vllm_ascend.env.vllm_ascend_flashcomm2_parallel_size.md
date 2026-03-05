---
topic_id: vllm_ascend.env.vllm_ascend_flashcomm2_parallel_size
canonical_term: VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE
topic_kind: parameter
---

# VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE

## Core

- topic_id: `vllm_ascend.env.vllm_ascend_flashcomm2_parallel_size`
- canonical_term: `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `throughput_tuning`
- status/confidence: `aligned` / `0.95`
- semantics: 调度和批处理参数调优，目标提升吞吐。
- aliases: `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE`, `vllm_ascend_flashcomm2_parallel_size`, `vllm-ascend-flashcomm2-parallel-size`, `vllm ascend flashcomm2 parallel size`, `throughput_tuning`, `throughput tuning`, `throughput-tuning`

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

- definition_ref: vllm_ascend/envs.py:79
- read_ref: vllm-ascend/vllm_ascend/envs.py:79, vllm-ascend/vllm_ascend/envs.py:79, vllm-ascend/vllm_ascend/utils.py:973
- effect_ref: vllm-ascend/vllm_ascend/utils.py:973
- web_refs: 4

## Details/Edge Cases

- failure_modes: TTFT/TPOT 退化; OOM
- value_failure_signals: TTFT/TPOT 退化; OOM
- recommendation: 按 TTFT/TPOT/吞吐三指标联合调参。
- updated_at: 2026-03-05
