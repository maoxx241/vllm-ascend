# Deployment decision report

- case_id: `case_b47c9c4ae1`
- result_class: `compatible`

## User request

本地有权重，A3，请求长度大概平均在3.5k输入1.5k输出，最大上下文40k，想要高吞吐

## Resolved subject

```json
{
  "model_name": "qwen3",
  "model_size_b": 32,
  "hardware": "A3",
  "cards": 8,
  "quantization": null,
  "weight_path": "/weights/qwen3-32b",
  "objective": "throughput"
}
```

## Assumptions

- 默认按单实例处理。
- A3 单机默认 8 卡 / 16 芯，不额外追问卡数。

## Why not exact

- 仓中的最强直接证据更接近 TP4 单实例配方；当前 8 卡单实例方案是基于该配方叠加 DP 的派生。

## Derived metrics

```json
{
  "avg_in": 3500,
  "avg_out": 1500,
  "max_context": 40000
}
```

## Reasoning sections

### intent_analysis

根据平均 3500 输入 / 1500 输出、最大上下文 40000 和高吞吐目标，优先按常规吞吐服务而不是 128K 长序列特殊路径处理。

### topology_reasoning

采用 TP4 作为核心形状，再在单实例内叠加 DP2；不直接假设 TP8 是最优。

## Launch candidates

### primary_tp4_dp2_throughput

- risk_level: `medium`

```bash
vllm serve /weights/qwen3-32b --served-model-name qwen3 --trust-remote-code --async-scheduling --distributed-executor-backend mp --tensor-parallel-size 4 --data-parallel-size 2 --max-model-len 40000 --max-num-batched-tokens 40000 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --additional-config '{"weight_prefetch_config":{"enabled":true}}' --gpu-memory-utilization 0.9 --block-size 128
```

Environment:
- `TASK_QUEUE_ENABLE=1`
- `HCCL_OP_EXPANSION_MODE=AIV`
- `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`

Rationale:
- TP4 is the strongest documented core shape for Qwen3-32B throughput serving.
- Single-instance 8-card serving prefers TP4 + DP2 over assuming TP8 is optimal.

### tp4_dp2_graph_capture

- risk_level: `medium`

```bash
vllm serve /weights/qwen3-32b --served-model-name qwen3 --trust-remote-code --async-scheduling --distributed-executor-backend mp --tensor-parallel-size 4 --data-parallel-size 2 --max-model-len 40000 --max-num-batched-tokens 40000 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,12,16,20,24,32,48,60,64,68,72,76,80]}' --additional-config '{"weight_prefetch_config":{"enabled":true}}' --gpu-memory-utilization 0.9 --block-size 128
```

Environment:
- `TASK_QUEUE_ENABLE=1`
- `HCCL_OP_EXPANSION_MODE=AIV`
- `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`

Rationale:
- Nightly configs suggest graph capture sizes for feature-stack serving on A3.
- Use this only after the primary script starts cleanly.

