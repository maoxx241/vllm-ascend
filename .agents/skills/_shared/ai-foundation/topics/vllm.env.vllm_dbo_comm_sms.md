---
topic_id: vllm.env.vllm_dbo_comm_sms
canonical_term: VLLM_DBO_COMM_SMS
topic_kind: parameter
---

# VLLM_DBO_COMM_SMS

## Core

- topic_id: `vllm.env.vllm_dbo_comm_sms`
- canonical_term: `VLLM_DBO_COMM_SMS`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `throughput_tuning`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: 调度和批处理参数调优，目标提升吞吐。
- aliases: `VLLM_DBO_COMM_SMS`, `vllm_dbo_comm_sms`, `vllm-dbo-comm-sms`, `vllm dbo comm sms`, `throughput_tuning`, `throughput tuning`, `throughput-tuning`

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

- definition_ref: vllm/envs.py:1484
- read_ref: vllm/vllm/envs.py:217, vllm/vllm/envs.py:1484, vllm/vllm/envs.py:1484
- effect_ref: vllm/vllm/envs.py:217, vllm/vllm/envs.py:1484, vllm/vllm/envs.py:1484
- web_refs: 2

## Details/Edge Cases

- failure_modes: TTFT/TPOT 退化; OOM
- value_failure_signals: TTFT/TPOT 退化; OOM
- recommendation: 按 TTFT/TPOT/吞吐三指标联合调参。
- updated_at: 2026-03-11
