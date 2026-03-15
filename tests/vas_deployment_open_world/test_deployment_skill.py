from __future__ import annotations

from pathlib import Path
import textwrap

from tools.vas_deployment_open_world.parser import parse_request
from tools.vas_deployment_open_world.compiler import EvidenceCompiler
from tools.vas_deployment_open_world.engine import DeploymentEngine


def make_stub_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'vllm-ascend'
    (repo / 'docs/source/tutorials/models').mkdir(parents=True, exist_ok=True)
    (repo / 'docs/source/tutorials/hardwares').mkdir(parents=True, exist_ok=True)
    (repo / 'docs/source/user_guide/support_matrix').mkdir(parents=True, exist_ok=True)
    (repo / 'docs/source').mkdir(parents=True, exist_ok=True)
    (repo / 'tests/e2e/nightly/single_node/models/configs').mkdir(parents=True, exist_ok=True)
    (repo / 'vllm_ascend/patch/worker').mkdir(parents=True, exist_ok=True)
    sibling = tmp_path / 'vllm'
    (sibling / 'vllm/model_executor/models').mkdir(parents=True, exist_ok=True)

    (repo / 'docs/source/user_guide/support_matrix/supported_models.md').write_text(textwrap.dedent('''
    | GLM-4.x | ✅ | A2/A3 |
    | Qwen3 | ✅ | A2/A3 |
    | Qwen2-VL | ✅ | A2/A3 |
    | DeepSeek V3/3.1 | ✅ | A2/A3 |
    '''), encoding='utf-8')
    (repo / 'docs/source/tutorials/models/GLM4.x.md').write_text(textwrap.dedent('''
    GLM-4.5/4.6/4.7
    --tensor-parallel-size 16
    For single-node deployment, we recommend using dp1tp16 and turn off expert parallel in low-latency scenarios.
    '''), encoding='utf-8')
    (repo / 'tests/e2e/nightly/single_node/models/configs/GLM-4.5.yaml').write_text(textwrap.dedent('''
    --enable-expert-parallel
    GLM-4.5-TP8-DP2-fullgraph
    '''), encoding='utf-8')
    (repo / 'docs/source/tutorials/models/Qwen3-Dense.md').write_text(textwrap.dedent('''
    Qwen3-32B-W8A8
    --tensor-parallel-size 4
    3.5K and an output of 1.5K
    '''), encoding='utf-8')
    (repo / 'docs/source/tutorials/models/DeepSeek-V3.1.md').write_text('require at least 2 Atlas 800 A2 (64G × 8)', encoding='utf-8')
    (repo / 'docs/source/tutorials/hardwares/310p.md').write_text('only supports eager mode and the float16 data type\nset `max-model-len` to a small value', encoding='utf-8')
    (repo / 'docs/source/faqs.md').write_text('Currently, w8a8, w4a8, and w4a4 quantization methods are already supported by vllm-ascend.', encoding='utf-8')
    (repo / 'vllm_ascend/patch/worker/patch_qwen3_5.py').write_text('# patch', encoding='utf-8')
    (sibling / 'vllm/model_executor/models/qwen3_5.py').write_text('# model', encoding='utf-8')
    return repo


def build_engine(repo: Path) -> DeploymentEngine:
    compiler = EvidenceCompiler(repo)
    evidence, recipes = compiler.compile()
    return DeploymentEngine(evidence, recipes)


def test_glm_requires_sla_and_lengths(tmp_path: Path):
    repo = make_stub_repo(tmp_path)
    engine = build_engine(repo)
    req = parse_request('我想要一个glm4.6 的A3单机的部署命令，权重在 /weights/glm46')
    result = engine.evaluate(req)
    assert result.result_class == 'needs_alignment'
    joined = '\n'.join(result.required_questions)
    assert 'TPOT' in joined
    assert '平均输入长度' in joined


def test_glm_low_latency_route(tmp_path: Path):
    repo = make_stub_repo(tmp_path)
    engine = build_engine(repo)
    req = parse_request('我想要一个glm4.6 的A3单机的部署命令，权重在 /weights/glm46-w8a8，平均输入4k输出1k，最大上下文8k，TPOT 20ms，W8A8')
    result = engine.evaluate(req)
    assert result.result_class == 'exact_verified'
    assert result.launch_candidates[0]['topology']['tp'] == 16
    assert result.launch_candidates[0]['topology']['ep'] is False


def test_glm_throughput_route(tmp_path: Path):
    repo = make_stub_repo(tmp_path)
    engine = build_engine(repo)
    req = parse_request('我想要一个glm4.6 的A3单机的部署命令，权重在 /weights/glm46-w8a8，平均输入4k输出1k，最大上下文8k，TPOT 80ms，W8A8')
    result = engine.evaluate(req)
    assert result.result_class == 'compatible'
    assert result.launch_candidates[0]['topology']['tp'] == 8
    assert result.launch_candidates[0]['topology']['dp'] == 2


def test_deepseek_resource_block(tmp_path: Path):
    repo = make_stub_repo(tmp_path)
    engine = build_engine(repo)
    req = parse_request('给我一个A2 双卡部署deepseek v3.1 的脚本，权重在 /weights/dsv31，平均输入4k输出1k，最大上下文16k，TPOT 80ms，BF16')
    result = engine.evaluate(req)
    assert result.result_class == 'blocked.resource'


def test_qwen2_vl_existing_weights_is_candidate(tmp_path: Path):
    repo = make_stub_repo(tmp_path)
    engine = build_engine(repo)
    req = parse_request('给我一个qwen2 vl 72b w4a4 的部署脚本，权重已在本地 /weights/qwen2-vl-w4a4，A3，平均输入2k输出500，最大上下文8k，TPOT 80ms')
    result = engine.evaluate(req)
    assert result.result_class == 'candidate'
