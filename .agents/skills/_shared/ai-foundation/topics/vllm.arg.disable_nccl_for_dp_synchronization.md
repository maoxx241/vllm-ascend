---
topic_id: vllm.arg.disable_nccl_for_dp_synchronization
canonical_term: --disable-nccl-for-dp-synchronization
topic_kind: parameter
---

# --disable-nccl-for-dp-synchronization

## Core

- topic_id: `vllm.arg.disable_nccl_for_dp_synchronization`
- canonical_term: `--disable-nccl-for-dp-synchronization`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `data_parallel`
- status/confidence: `needs_manual_review` / `0.79`
- semantics: 通过副本扩展吞吐能力，并依赖 DP 地址和 RPC 协调。
- aliases: `--disable-nccl-for-dp-synchronization`, `disable-nccl-for-dp-synchronization`, `disable_nccl_for_dp_synchronization`, `disable nccl for dp synchronization`, `disablencclfordpsynchronization`, `data_parallel`, `data parallel`, `data-parallel`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `data_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 默认 unset(None)：async scheduling 开启时自动设为 True，否则 False。
- value_shape: `binary_or_auto`
- accepted_values: enabled, disabled, unset(auto)
- constraints: 该项主要影响 DP 同步实现，非 DP 场景影响有限。
- combo_effects: 与 --async-scheduling、--data-parallel-size 联动最明显。

## Development View

- definition_ref: vllm/engine/arg_utils.py:891
- read_ref: vllm/vllm/config/parallel.py:185, vllm/vllm/config/parallel.py:298, vllm/vllm/config/vllm.py:694
- effect_ref: vllm/vllm/config/vllm.py:694, vllm/vllm/v1/worker/dp_utils.py:29, vllm/vllm/engine/arg_utils.py:892
- web_refs: 4

## Details/Edge Cases

- failure_modes: RPC 连接失败; 请求分发不均衡
- value_failure_signals: 通信栈不匹配时可能出现同步性能下降或超时。
- recommendation: 固定 DP 地址和端口后再迭代性能参数。
- updated_at: 2026-03-05
