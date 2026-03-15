from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
TOOLS_ROOT = REPO_ROOT / 'tools'
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from vas_deployment_skill.assistant_entry import vllm_ascend_assistant
from vas_deployment_skill.compiler import OpenWorldCompiler
from vas_deployment_skill.engine import evaluate_text

BUILD_DIR = REPO_ROOT / '.vas_test_build'
RUNTIME_WS = REPO_ROOT / '.vas_test_cases'


def _ensure_build() -> None:
    if not (BUILD_DIR / 'evidence.jsonl').exists():
        OpenWorldCompiler(WORKSPACE_ROOT).compile(BUILD_DIR)


def test_qwen3_32b_turn1_requires_user_only_facts() -> None:
    _ensure_build()
    result = evaluate_text(BUILD_DIR, '给我一个qwen3 32b 8卡的部署命令')
    assert result.result_class == 'blocked.user_only_fact'


def test_qwen2_vl_existing_w4a4_weights_returns_candidate() -> None:
    _ensure_build()
    result = evaluate_text(
        BUILD_DIR,
        '给我一个qwen2 vl 72b w4a4的部署脚本',
        overrides={
            'hardware': 'A3',
            'model_name': 'qwen2-vl',
            'model_size_b': 72,
            'quantization': 'W4A4',
            'weight_path': '/weights/qwen2-vl-72b-w4a4',
            'has_existing_quantized_weights': True,
            'max_context': 4096,
        },
    )
    assert result.result_class == 'candidate'


def test_entry_writes_bundle() -> None:
    _ensure_build()
    payload = vllm_ascend_assistant(
        workspace_root=RUNTIME_WS,
        build_dir=BUILD_DIR,
        text='本地有权重，A3，请求长度大概平均在3.5k输入1.5k输出，最大上下文40k，想要高吞吐',
        overrides={
            'model_name': 'qwen3',
            'model_size_b': 32,
            'hardware': 'A3',
            'cards': 8,
            'weight_path': '/weights/qwen3-32b',
            'average_input_len': 3500,
            'average_output_len': 1500,
            'max_context': 40000,
            'objective': 'throughput',
            'local_weights': True,
        },
    )
    assert payload['route'] == 'deployment-synthesis'
