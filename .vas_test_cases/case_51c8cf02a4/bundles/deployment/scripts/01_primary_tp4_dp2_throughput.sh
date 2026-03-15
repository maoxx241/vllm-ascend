#!/usr/bin/env bash
set -euo pipefail

# candidate: primary_tp4_dp2_throughput

export TASK_QUEUE_ENABLE="1"
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_ASCEND_ENABLE_FLASHCOMM1="1"

vllm serve /weights/qwen3-32b --served-model-name qwen3 --trust-remote-code --async-scheduling --distributed-executor-backend mp --tensor-parallel-size 4 --data-parallel-size 2 --max-model-len 40000 --max-num-batched-tokens 40000 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --additional-config '{"weight_prefetch_config":{"enabled":true}}' --gpu-memory-utilization 0.9 --block-size 128
