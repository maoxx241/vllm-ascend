# SPDX-License-Identifier: Apache-2.0
"""Check replicated query geometry against independent TP-sharded projections."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def load_impl():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/context_parallel/mla_cp.py"
    tree = ast.parse(path.read_text())
    impl = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendMlaDCPImpl")
    impl.bases = []
    impl.body = [
        n
        for n in impl.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"dcp_q_replicate", "_project_query", "_q_proj_and_k_up_proj", "reorg_decode_q"}
    ]
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[impl], type_ignores=[]), str(path), "exec"), scope)
    return scope["AscendMlaDCPImpl"]


@pytest.mark.parametrize("dcp_size", [2, 4, 8])
@pytest.mark.parametrize("tokens", [1, 4, 16])
def test_replicated_absorption_matches_sharded_projection(dcp_size, tokens):
    torch.manual_seed(42)
    heads, nope, rope, latent, hidden = 3, 8, 4, 16, 12
    q_weight = torch.randn(dcp_size * heads * (nope + rope), hidden)
    k_weight = torch.randn(dcp_size * heads, nope, latent)
    x = torch.randn(tokens, hidden)
    expected_abs, expected_pe = [], []
    for rank in range(dcp_size):
        wq = q_weight.chunk(dcp_size)[rank]
        q = (x @ wq.T).view(tokens, heads, nope + rope)
        wk = k_weight.chunk(dcp_size)[rank]
        expected_abs.append(torch.einsum("thd,hdl->thl", q[..., :nope], wk))
        expected_pe.append(q[..., nope:])
    expected_abs = torch.cat(expected_abs, dim=1)
    expected_pe = torch.cat(expected_pe, dim=1)

    class Projection:
        qrep_active = True

        def __call__(self, value):
            return value @ q_weight.T, None

        def _local_view(self, value):
            return value[:, rank * heads : (rank + 1) * heads].contiguous()

    impl = load_impl()()
    impl.num_heads, impl.dcp_size = heads, dcp_size
    impl.qk_nope_head_dim, impl.qk_rope_head_dim = nope, rope
    impl.qk_head_dim = nope + rope
    impl.q_proj = Projection()
    impl.dcp_W_UK_T = k_weight
    impl._dcp_all_gather_fragments = lambda *args, **kwargs: pytest.fail("runtime Q all-gather")
    actual_abs, actual_pe = impl._q_proj_and_k_up_proj(x)
    actual_abs, actual_pe = impl.reorg_decode_q(actual_abs, actual_pe)
    torch.testing.assert_close(actual_abs, expected_abs)
    torch.testing.assert_close(actual_pe, expected_pe)
    for rank in range(dcp_size):
        local = impl._project_query(x, local_heads=True)
        expected = (x @ q_weight.chunk(dcp_size)[rank].T).view(tokens, heads, nope + rope)
        torch.testing.assert_close(local, expected)


@pytest.mark.parametrize("consumer", [False, True])
def test_pd_last_prompt_token_classification(consumer):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/dcp_utils.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DCPManager")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "classify_decode_request_mask")
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    actual = scope[method.name](
        SimpleNamespace(is_pd_decode_node=consumer),
        torch.tensor([1, 4, 1, 5, 1, 1]),
        torch.tensor([9, 9, 8, 9, 10, 0]),
        torch.tensor([10, 10, 10, 10, 10, 1]),
        4,
    )
    assert actual.tolist() == [consumer, consumer, False, False, True, consumer]
