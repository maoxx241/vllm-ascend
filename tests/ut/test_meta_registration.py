# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

_HIGH_RISK_META_PATTERNS = (
    ".size(",
    ".sizes(",
    ".numel(",
    "at::empty({",
    "torch::empty({",
    "at::zeros({",
    "torch::zeros({",
    "empty_symint({",
    "std::vector<int64_t>",
    "at::SmallVector<int64_t",
    "c10::SmallVector<int64_t",
)
_EXEMPTION_MARKER = "symbolic-meta-ok:"


def _has_symbolic_meta_exemption(lines: list[str], line_index: int) -> bool:
    candidates = lines[max(0, line_index - 1) : line_index + 1]
    return any(
        (marker_index := candidate.find(_EXEMPTION_MARKER)) >= 0
        and candidate[marker_index + len(_EXEMPTION_MARKER) :].strip()
        for candidate in candidates
    )


def test_kda_meta_kernels_preserve_symbolic_shapes():
    """Keep KDA additions compatible with the upstream symbolic-meta gate."""
    source_path = Path(__file__).parents[2] / "csrc" / "torch_binding_meta.cpp"
    source = source_path.read_text(encoding="utf-8")
    start = source.index("chunk_kda_fwd_meta(")
    end = source.index("\nvoid store_kv_block_metadata(", start)
    lines = source[start:end].splitlines()

    violations = [
        (line_index + 1, pattern)
        for line_index, line in enumerate(lines)
        if not _has_symbolic_meta_exemption(lines, line_index)
        for pattern in _HIGH_RISK_META_PATTERNS
        if pattern in line
    ]

    assert not violations, f"KDA meta kernels materialize symbolic shapes: {violations}"
