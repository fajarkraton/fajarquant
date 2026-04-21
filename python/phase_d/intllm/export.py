"""IntLLM .fjm v9 exporter — PyTorch checkpoint → FajarOS-loadable file.

Implements the byte layout described in `docs/FJQ_PHASE_D_FJM_V9_SPEC.md`.

**Spec correction discovered while writing this exporter:** the v9 spec
§4 originally listed 7 BitLinears per layer (4 MLGRU + 3 GLU per the
MatMul-Free paper), but upstream `ridgerchu/matmulfreellm @ f24cfe5`
ships HGRN-Bit with **6 BitLinears per layer**:

  attn.i_proj  : (d, d)      ← MLGRU input
  attn.f_proj  : (d, d)      ← MLGRU forget
  attn.g_proj  : (d, d)      ← MLGRU gate
  attn.o_proj  : (d, d)      ← MLGRU output
  mlp.gate_proj: (2·int, d)  ← GLU combined gate+up (memory-optimized)
  mlp.down_proj: (d, int)    ← GLU down

Plus 1 LM head BitLinear of shape (V, d).

Total per network: `6L + 1` BitLinears. v9 spec §4 should be updated;
this exporter sources from the actual model layout, not the spec text.
The v9 binary format is unchanged — only the indexing convention shifts.

Usage:
    from intllm.model import HGRNBitForCausalLM
    from intllm.export import export_fjm_v9
    model = HGRNBitForCausalLM.from_pretrained('ridger/MMfreeLM-370M')
    export_fjm_v9(model, '/tmp/intllm_370m.fjm')
"""

from __future__ import annotations

import struct
from pathlib import Path

import torch
import torch.nn as nn

# v9 protocol constants
FJM_MAGIC = b"FJM\x00"
FJM_VERSION = 9
MODEL_TYPE_INTLLM = 11   # MatMul-Free LM
QUANT_FORMAT_TERNARY = 2  # BitNet absmean
HEADER_SIZE = 176         # matches v7/v8 skeleton

# Per-layer block header (24 bytes per v9 spec §4)
LAYER_HDR_SIZE = 24


def _is_bitlinear(module: nn.Module) -> bool:
    return module.__class__.__name__ in {"BitLinear", "FusedBitLinear", "BitLinear_wonorm_bmm"}


def _absmean_beta(weight: torch.Tensor) -> float:
    """Compute the per-matrix absmean β scalar (BitNet convention)."""
    return float(weight.abs().mean().item())


def quantize_ternary(weight: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, float]:
    """Quantize a weight tensor to ternary {-1, 0, +1} via absmean.

    Returns `(quantized_int8, beta)` where:
      - quantized_int8 is a tensor of dtype torch.int8 holding {-1, 0, +1}
      - beta is the FP32 absmean scale needed at inference time
    """
    beta = _absmean_beta(weight) + eps
    scale = 1.0 / beta
    q = (weight.detach() * scale).round().clamp(-1, 1).to(torch.int8)
    return q, beta


def pack_ternary(q_int8: torch.Tensor) -> bytes:
    """Pack a ternary {-1, 0, +1} int8 tensor into 2-bit-per-entry bytes.

    Encoding (v9 §5):
      00 = -1
      01 =  0
      10 = +1
      11 = reserved (never written)

    Layout: row-major over (out_features, in_features), 4 entries per
    byte little-endian.
    """
    # Map {-1, 0, +1} → {0, 1, 2} as uint8
    flat = q_int8.flatten().to(torch.int64)
    coded = (flat + 1).to(torch.uint8)  # -1→0, 0→1, +1→2
    n = coded.numel()
    n_bytes = (n + 3) // 4
    out = bytearray(n_bytes)
    # Build by groups of 4
    for i in range(n_bytes):
        b = 0
        for j in range(4):
            idx = i * 4 + j
            if idx < n:
                b |= (int(coded[idx]) & 0x3) << (2 * j)
        out[i] = b
    return bytes(out)


def _enumerate_bitlinears(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Find all BitLinear-family modules in deterministic
    state_dict-order."""
    return [(name, mod) for name, mod in model.named_modules() if _is_bitlinear(mod)]


def _build_header(*, n_layers: int, d_model: int, vocab: int, n_heads: int = 1) -> bytes:
    """Build the 176-byte header for a v9 file.

    Mirrors the v7/v8 layout but with v9 semantics:
      version = 9, model_type = 11, quant_format = 2,
      rope_global = 0, sliding_window = 0, sliding_pattern = 0
    """
    h = bytearray(HEADER_SIZE)
    h[0:4] = FJM_MAGIC
    h[4] = FJM_VERSION
    h[5] = MODEL_TYPE_INTLLM
    # Standard fields used by all readers (offsets per fajaros-x86 model_loader.fj)
    struct.pack_into("<I", h, 8, n_layers)
    struct.pack_into("<I", h, 12, d_model)
    struct.pack_into("<I", h, 16, vocab)
    struct.pack_into("<I", h, 20, n_heads)
    # v7/v8 fields at 152-175
    struct.pack_into("<Q", h, 152, 0)   # rope_global = 0 (NoPE)
    struct.pack_into("<I", h, 160, 0)   # sliding_window = 0
    struct.pack_into("<I", h, 164, 0)   # sliding_pattern = 0
    struct.pack_into("<I", h, 168, n_heads)  # n_kv_heads_v7 (== n_heads for HGRN)
    struct.pack_into("<H", h, 172, QUANT_FORMAT_TERNARY)
    struct.pack_into("<H", h, 174, 0)   # group_size = 0 (per-matrix β, not group-wise)
    return bytes(h)


def _build_beta_table(betas: list[float]) -> bytes:
    """β table per v9 §2: u32 table_size, u32 n_betas, then f32 entries."""
    n = len(betas)
    out = bytearray()
    out.extend(struct.pack("<I", 4 * n))   # table_size in bytes
    out.extend(struct.pack("<I", n))        # n_betas
    for beta in betas:
        out.extend(struct.pack("<f", beta))
    return bytes(out)


def _build_empty_gamma_table() -> bytes:
    """Empty γ_x table — exporter writes header but no entries when periodic
    re-cal QAT was not enabled. Reader interprets n_entries=0 as 'compute
    γ_x per-call' (BitNet baseline)."""
    out = bytearray()
    out.extend(struct.pack("<I", 0))   # table_size = 0
    out.extend(struct.pack("<I", 0))   # n_entries = 0
    return bytes(out)


def _build_layer_block(layer_id: int, layer_modules: dict[str, nn.Module]) -> tuple[bytes, list[float]]:
    """Build one layer block per v9 §4. Returns (block_bytes, layer_betas)."""
    # Order matches v9 spec indexing: i, f, g, o (attn) + gate, down (mlp)
    proj_order = ["attn.i_proj", "attn.f_proj", "attn.g_proj", "attn.o_proj",
                  "mlp.gate_proj", "mlp.down_proj"]
    weight_bytes = bytearray()
    betas: list[float] = []
    mlgru_size = 0
    glu_size = 0
    for proj_name in proj_order:
        mod = layer_modules.get(proj_name)
        if mod is None:
            raise ValueError(f"layer {layer_id} missing {proj_name}")
        q, beta = quantize_ternary(mod.weight)
        packed = pack_ternary(q)
        if proj_name.startswith("attn."):
            mlgru_size += len(packed)
        else:
            glu_size += len(packed)
        weight_bytes.extend(packed)
        betas.append(beta)

    # 24-byte layer header
    rmsnorm_offset = LAYER_HDR_SIZE + len(weight_bytes)
    rmsnorm_eps = 1e-6  # FJQ_PHASE_D_OPS.md §2 Op 2
    total_size = LAYER_HDR_SIZE + len(weight_bytes) + 4  # +4 for FP32 eps

    hdr = bytearray(LAYER_HDR_SIZE)
    struct.pack_into("<I", hdr, 0, layer_id)
    struct.pack_into("<I", hdr, 4, total_size)
    struct.pack_into("<I", hdr, 8, mlgru_size)
    struct.pack_into("<I", hdr, 12, glu_size)
    struct.pack_into("<I", hdr, 16, rmsnorm_offset)
    struct.pack_into("<I", hdr, 20, 0)  # reserved

    block = bytes(hdr) + bytes(weight_bytes) + struct.pack("<f", rmsnorm_eps)
    return block, betas


def export_fjm_v9(model: nn.Module, out_path: str | Path) -> dict:
    """Export an HGRN-Bit (Phase D-style) PyTorch model to .fjm v9.

    Writes the file at `out_path` and returns a manifest dict with byte
    offsets + sizes — useful for sidecar diagnostic JSON.
    """
    out_path = Path(out_path)
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    d_model = cfg.hidden_size
    vocab = cfg.vocab_size

    bitlinears = _enumerate_bitlinears(model)
    if not bitlinears:
        raise ValueError("no BitLinear modules found in model")

    # Group by layer
    layers: dict[int, dict[str, nn.Module]] = {i: {} for i in range(n_layers)}
    lm_head: nn.Module | None = None
    for name, mod in bitlinears:
        if name.startswith("model.layers."):
            parts = name.split(".")
            layer_id = int(parts[2])
            sub = ".".join(parts[3:])  # e.g. "attn.i_proj"
            layers[layer_id][sub] = mod
        elif name == "lm_head":
            lm_head = mod
        else:
            # Unknown — skip with a warning embedded in the manifest
            continue
    if lm_head is None:
        raise ValueError("model has no lm_head BitLinear")

    # Build sections
    header = _build_header(n_layers=n_layers, d_model=d_model, vocab=vocab,
                           n_heads=getattr(cfg, "num_heads", 1))

    all_betas: list[float] = []
    layer_blocks: list[bytes] = []
    for i in range(n_layers):
        block, betas = _build_layer_block(i, layers[i])
        layer_blocks.append(block)
        all_betas.extend(betas)

    # LM head: ternary [V, d] in its own pseudo-layer block
    lm_q, lm_beta = quantize_ternary(lm_head.weight)
    lm_packed = pack_ternary(lm_q)
    all_betas.append(lm_beta)

    beta_table = _build_beta_table(all_betas)
    gamma_table = _build_empty_gamma_table()

    # Embedding (FP16, row-major) — pull from model.embeddings
    embed_module = model.get_input_embeddings()
    embed_bytes = embed_module.weight.detach().to(torch.float16).cpu().numpy().tobytes()

    # Compose final file
    parts = [header, beta_table, gamma_table, *layer_blocks, embed_bytes, lm_packed]
    blob = b"".join(parts)
    out_path.write_bytes(blob)

    # Manifest for sidecar JSON
    offset = 0
    manifest = {
        "path": str(out_path),
        "version": FJM_VERSION,
        "model_type": MODEL_TYPE_INTLLM,
        "quant_format": QUANT_FORMAT_TERNARY,
        "n_layers": n_layers,
        "d_model": d_model,
        "vocab_size": vocab,
        "n_betas": len(all_betas),
        "total_bytes": len(blob),
        "sections": [],
    }
    for label, data in [
        ("header", header),
        ("beta_table", beta_table),
        ("gamma_table", gamma_table),
        *[(f"layer_{i}", b) for i, b in enumerate(layer_blocks)],
        ("embedding", embed_bytes),
        ("lm_head", lm_packed),
    ]:
        manifest["sections"].append({"label": label, "offset": offset, "size": len(data)})
        offset += len(data)
    return manifest


def parse_fjm_v9_header(path: str | Path) -> dict:
    """Read the v9 header back from a .fjm file.

    Companion to `export_fjm_v9` — used by tests + by the reader-rejection
    safety check in §6.8 R3 (verify v8 readers reject v9 cleanly).
    """
    data = Path(path).read_bytes()
    if data[:4] != FJM_MAGIC:
        raise ValueError(f"bad magic: {data[:4]!r}")
    version = data[4]
    model_type = data[5]
    n_layers, = struct.unpack_from("<I", data, 8)
    d_model,  = struct.unpack_from("<I", data, 12)
    vocab,    = struct.unpack_from("<I", data, 16)
    n_heads,  = struct.unpack_from("<I", data, 20)
    quant_format, = struct.unpack_from("<H", data, 172)
    return {
        "version": version,
        "model_type": model_type,
        "n_layers": n_layers,
        "d_model": d_model,
        "vocab_size": vocab,
        "n_heads": n_heads,
        "quant_format": quant_format,
    }


__all__ = [
    "FJM_MAGIC",
    "FJM_VERSION",
    "MODEL_TYPE_INTLLM",
    "QUANT_FORMAT_TERNARY",
    "export_fjm_v9",
    "pack_ternary",
    "parse_fjm_v9_header",
    "quantize_ternary",
]
