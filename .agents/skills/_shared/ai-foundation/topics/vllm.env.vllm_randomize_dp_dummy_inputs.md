---
topic_id: vllm.env.vllm_randomize_dp_dummy_inputs
canonical_term: VLLM_RANDOMIZE_DP_DUMMY_INPUTS
topic_kind: parameter
---

# VLLM_RANDOMIZE_DP_DUMMY_INPUTS

## Core

- topic_id: `vllm.env.vllm_randomize_dp_dummy_inputs`
- canonical_term: `VLLM_RANDOMIZE_DP_DUMMY_INPUTS`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `data_parallel`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: 通过副本扩展吞吐能力，并依赖 DP 地址和 RPC 协调。
- aliases: `VLLM_RANDOMIZE_DP_DUMMY_INPUTS`, `vllm_randomize_dp_dummy_inputs`, `vllm-randomize-dp-dummy-inputs`, `vllm randomize dp dummy inputs`, `data_parallel`, `data parallel`, `data-parallel`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `data_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 错误地址/端口会导致调度与健康检查失败
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:1062
- read_ref: vllm/vllm/envs.py:139, vllm/vllm/envs.py:1062, vllm/vllm/envs.py:1063
- effect_ref: vllm/vllm/v1/worker/gpu_model_runner.py:4540
- web_refs: 3

## Details/Edge Cases

- failure_modes: RPC 连接失败; 请求分发不均衡
- value_failure_signals: RPC 连接失败; 请求分发不均衡
- recommendation: 固定 DP 地址和端口后再迭代性能参数。
- updated_at: 2026-03-06
