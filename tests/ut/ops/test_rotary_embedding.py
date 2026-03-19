import torch

from vllm_ascend.ops import rotary_embedding as rope_mod
from vllm_ascend.ops.rotary_embedding import AscendMRotaryEmbedding


def test_maybe_all_gather_mtp_positions_uses_token_dim_for_mrope(monkeypatch):
    positions = torch.arange(3, dtype=torch.long).view(3, 1)
    expected = torch.arange(12, dtype=torch.long).view(3, 4)
    called = {}

    def fake_gather(x, pad, unpad, token_dim):
        called["shape"] = list(x.shape)
        called["pad"] = pad
        called["unpad"] = unpad
        called["token_dim"] = token_dim
        return expected

    monkeypatch.setattr(rope_mod._EXTRA_CTX, "is_draft_model", True, raising=False)
    monkeypatch.setattr(
        rope_mod._EXTRA_CTX,
        "flash_comm_v1_enabled",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "maybe_all_gather_and_maybe_unpad",
        fake_gather,
    )

    actual = rope_mod._maybe_all_gather_mtp_positions(positions, use_mtp=True)

    assert actual is expected
    assert called == {
        "shape": [3, 1],
        "pad": True,
        "unpad": False,
        "token_dim": 1,
    }


def test_mrotary_forward_triton_uses_gathered_positions(monkeypatch):
    positions = torch.zeros((3, 1), dtype=torch.long)
    gathered_positions = torch.zeros((3, 4), dtype=torch.long)
    query = torch.randn(4, 16, dtype=torch.float32)
    key = torch.randn(4, 8, dtype=torch.float32)
    rope = AscendMRotaryEmbedding(
        head_size=8,
        rotary_dim=6,
        max_position_embeddings=16,
        base=10000.0,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=[1, 1, 1],
        mrope_interleaved=True,
    )

    def fake_maybe_gather(x, use_mtp):
        assert use_mtp is True
        assert x is positions
        return gathered_positions

    def fake_triton_mrope(q, k, cos, sin, mrope_section, head_size, rotary_dim, mrope_interleaved):
        assert list(q.shape) == [4, 16]
        assert list(k.shape) == [4, 8]
        assert list(cos.shape[:2]) == [3, 4]
        assert list(sin.shape[:2]) == [3, 4]
        return q, k

    monkeypatch.setattr(rope, "use_mtp", True, raising=False)
    monkeypatch.setattr(
        rope_mod,
        "_maybe_all_gather_mtp_positions",
        fake_maybe_gather,
    )
    monkeypatch.setattr(rope_mod, "triton_mrope", fake_triton_mrope)

    q_out, k_out = rope.forward_triton(positions, query, key)

    assert q_out is query
    assert k_out is key
