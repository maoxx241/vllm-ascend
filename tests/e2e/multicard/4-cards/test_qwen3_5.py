#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/basic_correctness/test_basic_correctness.py
#
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from tests.e2e.conftest import VllmRunner, qwen_prompt

QWEN35_MTP3_SPECULATIVE_CONFIG = {
    "method": "qwen3_5_mtp",
    "num_speculative_tokens": 3,
}
QWEN35_GRAPH_COMPILATION_CONFIG = {
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [1, 2, 4, 8],
}


def _get_qwen35_mtp3_speculative_config():
    return dict(QWEN35_MTP3_SPECULATIVE_CONFIG)


def test_qwen3_5_27b_distributed_mp_tp4():
    example_prompts = [
        "Hello, my name is",
    ] * 4
    max_tokens = 5
    with VllmRunner("Qwen/Qwen3.5-27B",
                    tensor_parallel_size=4,
                    cudagraph_capture_sizes=[1, 2, 4, 8],
                    max_model_len=4096,
                    gpu_memory_utilization=0.90,
                    distributed_executor_backend="mp",
                    speculative_config=_get_qwen35_mtp3_speculative_config()) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
        del vllm_model


@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "1"})
def test_qwen3_5_27b_distributed_mp_flash_comm_tp4():
    example_prompts = [
        "Hello, my name is",
    ] * 4
    max_tokens = 5
    model_name = os.environ.get("VLLM_QWEN35_TEXT_MODEL", "Qwen/Qwen3.5-27B")
    with VllmRunner(model_name,
                    tensor_parallel_size=4,
                    max_model_len=4096,
                    gpu_memory_utilization=0.90,
                    distributed_executor_backend="mp",
                    enforce_eager=True,
                    speculative_config=_get_qwen35_mtp3_speculative_config()) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
        del vllm_model


def test_qwen3_5_35b_distributed_mp_tp4():
    example_prompts = [
        "Hello, my name is",
    ] * 4
    max_tokens = 5
    with VllmRunner("Qwen/Qwen3.5-35B-A3B",
                    tensor_parallel_size=4,
                    cudagraph_capture_sizes=[1, 2, 4, 8],
                    max_model_len=4096,
                    gpu_memory_utilization=0.90,
                    distributed_executor_backend="mp",
                    speculative_config=_get_qwen35_mtp3_speculative_config()) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
        del vllm_model


def _get_qwen35_vl_test_image():
    image_path = Path(__file__).resolve().parents[2] / "310p" / "data" / "qwen.png"
    return Image.open(image_path).convert("RGB")


@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "1"})
def test_qwen3_5_local_vl_distributed_mp_flash_comm_tp2():
    model_name = os.environ.get("VLLM_QWEN35_VL_MODEL", "/home/weights/Qwen3.5-0.8B")
    if os.path.isabs(model_name) and not os.path.exists(model_name):
        pytest.skip(f"Local Qwen3.5 VL model path does not exist: {model_name}")

    image = _get_qwen35_vl_test_image()
    prompts = [
        "Briefly introduce yourself.",
        qwen_prompt(["Describe this image in detail."])[0],
    ]
    images = [None, image]

    with VllmRunner(model_name,
                    tensor_parallel_size=2,
                    max_model_len=512,
                    max_num_seqs=2,
                    gpu_memory_utilization=0.6,
                    distributed_executor_backend="mp",
                    enforce_eager=True,
                    dtype="bfloat16",
                    limit_mm_per_prompt={"image": 1},
                    speculative_config=_get_qwen35_mtp3_speculative_config()) as vllm_model:
        outputs = vllm_model.generate_greedy(prompts, max_tokens=32, images=images)

    assert len(outputs) == len(prompts)
    for _, output_str in outputs:
        assert output_str


@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "1"})
def test_qwen3_5_local_vl_distributed_mp_flash_comm_graph_mode_tp2():
    model_name = os.environ.get("VLLM_QWEN35_VL_MODEL", "/home/weights/Qwen3.5-0.8B")
    if os.path.isabs(model_name) and not os.path.exists(model_name):
        pytest.skip(f"Local Qwen3.5 VL model path does not exist: {model_name}")

    prompts = [
        "Briefly introduce yourself.",
        "Describe the future of AI in one paragraph.",
    ]

    with VllmRunner(model_name,
                    tensor_parallel_size=2,
                    max_model_len=512,
                    max_num_seqs=2,
                    gpu_memory_utilization=0.6,
                    distributed_executor_backend="mp",
                    enforce_eager=False,
                    dtype="bfloat16",
                    compilation_config=QWEN35_GRAPH_COMPILATION_CONFIG,
                    speculative_config=_get_qwen35_mtp3_speculative_config()) as vllm_model:
        outputs = vllm_model.generate_greedy(prompts, max_tokens=32)

    assert len(outputs) == len(prompts)
    for _, output_str in outputs:
        assert output_str


def test_qwen3_5_35b_distributed_mp_tp4_full_decode_only_mtp3():
    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    max_tokens = 20
    with VllmRunner("Qwen/Qwen3.5-35B-A3B",
                    tensor_parallel_size=4,
                    max_model_len=4096,
                    gpu_memory_utilization=0.90,
                    distributed_executor_backend="mp",
                    compilation_config={
                        "cudagraph_mode": "FULL_DECODE_ONLY",
                        "cudagraph_capture_sizes": [4, 8, 12, 16],
                    },
                    speculative_config=_get_qwen35_mtp3_speculative_config()) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
        del vllm_model
