#!/usr/bin/env bash
set -euo pipefail

# candidate: tp4_dp2_graph_capture

export TASK_QUEUE_ENABLE="1"
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_ASCEND_ENABLE_FLASHCOMM1="1"

vllm serve /weights/qwen3-32b --served-model-name qwen3 --trust-remote-code --async-scheduling --distributed-executor-backend mp --tensor-parallel-size 4 --data-parallel-size 2 --max-model-len 40000 --max-num-batched-tokens 40000 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,12,16,20,24,32,48,60,64,68,72,76,80]}' --additional-config '{"weight_prefetch_config":{"enabled":true}}' --gpu-memory-utilization 0.9 --block-size 128
