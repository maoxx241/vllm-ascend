---
topic_id: vllm_ascend.env.vllm_ascend_enable_flashcomm
canonical_term: VLLM_ASCEND_ENABLE_FLASHCOMM
topic_kind: parameter
---

# VLLM_ASCEND_ENABLE_FLASHCOMM

## Core

- topic_id: `vllm_ascend.env.vllm_ascend_enable_flashcomm`
- canonical_term: `VLLM_ASCEND_ENABLE_FLASHCOMM`
- kind/scope: `env` / `vllm_ascend`
- stage: `runtime`
- primary_feature: `throughput_tuning`
- status/confidence: `upstream_delta` / `0.75`
- source: `code` / source_tags: code_reference
- semantics: FlashComm1 旧兼容开关，推荐使用 VLLM_ASCEND_ENABLE_FLASHCOMM1。
- aliases: `VLLM_ASCEND_ENABLE_FLASHCOMM`, `vllm_ascend_enable_flashcomm`, `vllm-ascend-enable-flashcomm`, `vllm ascend enable flashcomm`, `throughput_tuning`, `throughput tuning`, `throughput-tuning`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `throughput_tuning` 查看稳定原理。

## Deployment View

- default_behavior: 默认 0（关闭），作为 FLASHCOMM1 兼容别名读取。
- value_shape: `binary_toggle`
- accepted_values: 0, 1
- constraints: 推荐改用 VLLM_ASCEND_ENABLE_FLASHCOMM1。
- combo_effects: 与 VLLM_ASCEND_ENABLE_FLASHCOMM1 同时设置时，以功能等价方式生效。

## Development View

- definition_ref: vllm_ascend/utils.py:765
- read_ref: vllm-ascend/vllm_ascend/utils.py:764, vllm-ascend/vllm_ascend/utils.py:765
- effect_ref: vllm-ascend/vllm_ascend/utils.py:764, vllm-ascend/vllm_ascend/utils.py:765
- web_refs: 4

## Details/Edge Cases

- failure_modes: TTFT/TPOT 退化; OOM
- value_failure_signals: 通信参数不匹配时收益不稳定或出现告警。
- recommendation: 按 TTFT/TPOT/吞吐三指标联合调参。
- updated_at: 2026-03-06
