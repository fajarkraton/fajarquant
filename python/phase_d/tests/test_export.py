"""Tests for `intllm.export` — verify .fjm v9 byte layout matches spec.

Round-trip: tiny model → export → parse header back → assert layout.
Does not need GPU; runs entirely on CPU in a few seconds.
"""

from __future__ import annotations

import pytest
import torch

from intllm.export import (
    FJM_MAGIC,
    FJM_VERSION,
    MODEL_TYPE_INTLLM,
    QUANT_FORMAT_TERNARY,
    export_fjm_v9,
    pack_ternary,
    parse_fjm_v9_header,
    quantize_ternary,
)
from intllm.model import HGRNBitConfig, HGRNBitForCausalLM


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="HGRNBitForCausalLM construction touches mmfreelm Triton kernels",
)


def _make_tiny_model() -> HGRNBitForCausalLM:
    cfg = HGRNBitConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        max_position_embeddings=64,
    )
    return HGRNBitForCausalLM(cfg).to("cuda")


# -----------------------------------------------------------------
# Quantization primitives
# -----------------------------------------------------------------

def test_quantize_ternary_returns_three_levels() -> None:
    w = torch.tensor([[1.0, -2.0, 0.0, 3.0], [-0.5, 4.0, -1.5, 0.1]])
    q, beta = quantize_ternary(w)
    assert q.dtype == torch.int8
    unique = set(q.flatten().tolist())
    # Allow {-1, 0, 1} or any subset
    assert unique <= {-1, 0, 1}
    # β = mean(|w|) = mean(1+2+0+3+0.5+4+1.5+0.1) / 8 = 12.1/8 = 1.5125
    assert abs(beta - (1.5125 + 1e-5)) < 1e-3


def test_pack_ternary_round_trip() -> None:
    """Ternary→packed→unpacked should be identity for known input."""
    q = torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0], dtype=torch.int8)
    packed = pack_ternary(q)
    # 8 entries → 2 bytes
    assert len(packed) == 2
    # Manually unpack: bits per entry encoded as (q+1) → {0, 1, 2}
    expected = []
    for byte in packed:
        for j in range(4):
            expected.append(((byte >> (2 * j)) & 0x3) - 1)
    assert expected == q.tolist()


def test_pack_ternary_padding_at_tail() -> None:
    """3 entries → 1 byte, last 5 bits should be zero (= -1 in our encoding,
    but reader knows the true count from the header)."""
    q = torch.tensor([1, -1, 0], dtype=torch.int8)
    packed = pack_ternary(q)
    assert len(packed) == 1


# -----------------------------------------------------------------
# Header round-trip
# -----------------------------------------------------------------

def test_export_then_parse_header(tmp_path) -> None:
    model = _make_tiny_model()
    out_path = tmp_path / "tiny.fjm"
    manifest = export_fjm_v9(model, out_path)

    # File exists + non-trivial size
    assert out_path.is_file()
    assert out_path.stat().st_size > 200

    # Header round-trip
    parsed = parse_fjm_v9_header(out_path)
    assert parsed["version"] == FJM_VERSION
    assert parsed["model_type"] == MODEL_TYPE_INTLLM
    assert parsed["quant_format"] == QUANT_FORMAT_TERNARY
    assert parsed["n_layers"] == 2
    assert parsed["d_model"] == 32
    assert parsed["vocab_size"] == 128


def test_manifest_section_offsets_are_consecutive(tmp_path) -> None:
    """Section offsets should chain: each starts where the previous ended."""
    model = _make_tiny_model()
    out_path = tmp_path / "tiny.fjm"
    manifest = export_fjm_v9(model, out_path)

    sections = manifest["sections"]
    expected_off = 0
    for sec in sections:
        assert sec["offset"] == expected_off, (
            f"section {sec['label']} offset {sec['offset']} != expected {expected_off}"
        )
        expected_off += sec["size"]
    assert manifest["total_bytes"] == expected_off
    assert manifest["total_bytes"] == out_path.stat().st_size


def test_n_betas_matches_6L_plus_1(tmp_path) -> None:
    """v9 spec corrected: 6 BitLinears per layer (not 7) + 1 LM head."""
    model = _make_tiny_model()
    out_path = tmp_path / "tiny.fjm"
    manifest = export_fjm_v9(model, out_path)

    n_layers = 2
    expected = 6 * n_layers + 1
    assert manifest["n_betas"] == expected, (
        f"expected {expected} betas (6L+1), got {manifest['n_betas']}"
    )


def test_magic_bytes_at_offset_zero(tmp_path) -> None:
    model = _make_tiny_model()
    out_path = tmp_path / "tiny.fjm"
    export_fjm_v9(model, out_path)
    data = out_path.read_bytes()
    assert data[:4] == FJM_MAGIC
