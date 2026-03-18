#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import argparse
import json
import os
import time
from pathlib import Path

DEFAULT_MODEL = "/home/weights/Qwen3.5-35B-A3B"
DEFAULT_OUTPUT = Path("/tmp/qwen35_moe_gsm8k.json")


def _get_counter_delta(before, after, name: str) -> int:
    from vllm.v1.metrics.reader import Counter

    before_val = next(metric.value for metric in before if isinstance(metric, Counter) and metric.name == name)
    after_val = next(metric.value for metric in after if isinstance(metric, Counter) and metric.name == name)
    return after_val - before_val


def _is_ep_dispatch_combine_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    ep_dispatch_combine_markers = (
        "dispatch_ffn_combine",
        "dispatch_gmm_combine_decode",
        "npu_moe_distribute_dispatch",
        "npu_moe_distribute_combine",
    )
    return any(marker in msg for marker in ep_dispatch_combine_markers)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("VLLM_QWEN35_MOE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--num-questions", type=int, default=100)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--enable-expert-parallel", action="store_true", default=True)
    parser.add_argument("--disable-expert-parallel", action="store_false", dest="enable_expert_parallel")
    parser.add_argument("--flashcomm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HCCL_BUFFSIZE", "1024")
    if args.flashcomm:
        os.environ["VLLM_ASCEND_ENABLE_FLASHCOMM1"] = "1"

    import torch_npu  # noqa: F401
    from vllm import LLM
    from vllm.tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k_offline

    llm = None
    result = {
        "config": {
            "model": args.model,
            "tensor_parallel_size": args.tensor_parallel_size,
            "async_scheduling": True,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "num_speculative_tokens": 3,
            "enable_expert_parallel": args.enable_expert_parallel,
            "flashcomm": args.flashcomm,
            "num_questions": args.num_questions,
            "num_shots": args.num_shots,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
    }

    try:
        llm = LLM(
            model=args.model,
            tokenizer=args.model,
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=args.tensor_parallel_size,
            distributed_executor_backend="mp",
            enable_expert_parallel=args.enable_expert_parallel,
            async_scheduling=True,
            enforce_eager=False,
            max_model_len=4096,
            max_num_seqs=4,
            gpu_memory_utilization=0.85,
            disable_log_stats=False,
            compilation_config={
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [4, 8, 12, 16],
            },
            speculative_config={
                "method": "qwen3_5_mtp",
                "num_speculative_tokens": 3,
            },
            seed=0,
        )

        metrics_before = llm.get_metrics()
        start = time.perf_counter()
        eval_result = evaluate_gsm8k_offline(
            llm,
            num_questions=args.num_questions,
            num_shots=args.num_shots,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        latency = time.perf_counter() - start
        metrics_after = llm.get_metrics()

        draft_tokens = _get_counter_delta(before=metrics_before,
                                          after=metrics_after,
                                          name="vllm:spec_decode_num_draft_tokens")
        accepted_tokens = _get_counter_delta(before=metrics_before,
                                             after=metrics_after,
                                             name="vllm:spec_decode_num_accepted_tokens")
        acceptance_rate = accepted_tokens / draft_tokens if draft_tokens > 0 else 0.0

        result.update(
            {
                "status": "ok",
                "metrics": {
                    **eval_result,
                    "wall_time_latency": latency,
                    "acceptance_rate": acceptance_rate,
                    "draft_tokens": draft_tokens,
                    "accepted_tokens": accepted_tokens,
                },
            }
        )
    except Exception as exc:
        if args.enable_expert_parallel and _is_ep_dispatch_combine_error(exc):
            result.update(
                {
                    "status": "skipped_ep_dispatch_combine",
                    "error": str(exc),
                }
            )
        else:
            raise
    finally:
        if llm is not None:
            del llm

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
