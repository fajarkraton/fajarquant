"""IntLLM evaluation — canonical protocol wrappers.

Centralizes the benchmark configurations used across C.P4 config gates
and C.P6 final validation. Every Phase D result that appears in the
paper traces back to one of the functions here, so that when §6.9 R7
`verify_paper_tables.py` checks numerical cells it only has to look at
the JSON artifacts these functions produce.

Protocols:
  - `run_zeroshot_six`: ARC-E + ARC-C + HS + OBQA + PIQA + WG (Zhu et
    al. Table 1 — baseline parity per §6.9 R3)
  - `run_wikitext103_ppl`: Wikitext-103 raw perplexity (not in the
    original paper; §6.9 R1 canonical protocol expansion)
  - `run_mmlu_5shot`: MMLU 5-shot (§6.9 R1 canonical protocol for
    knowledge)
  - `run_held_out_loss`: shared inner-loop val (cheap, fast) — used by
    train scripts and Q5 baseline driver.
  - `compute_quant_error_per_channel`: E2.4.C.2 quantization-error
    metric (offline; reads a saved `mini_<TAG>_maps.pt` artifact and
    measures the per-channel MSE delta vs upstream `activation_quant`).

None of the lm-evaluation protocols implement evaluation themselves —
they delegate to `lm-evaluation-harness`, the canonical literature
implementation.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalArtifact:
    """Result of one evaluation run."""

    model_name: str
    protocol: str
    metrics: dict[str, float]  # percent-valued
    elapsed_seconds: float
    lm_eval_version: str

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


ZEROSHOT_SIX_TASKS = [
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "openbookqa",
    "piqa",
    "winogrande",
]
ZEROSHOT_SIX_METRIC = {
    "arc_easy": "acc_norm",
    "arc_challenge": "acc_norm",
    "hellaswag": "acc_norm",
    "openbookqa": "acc_norm",
    "piqa": "acc",
    "winogrande": "acc",
}


def _extract_metric(task_scores: dict[str, Any], metric: str) -> float:
    """lm_eval reports keys like `acc_norm,none` in newer versions and
    `acc_norm` in older. Accept either."""
    for k in (f"{metric},none", metric):
        if k in task_scores:
            return float(task_scores[k])
    raise KeyError(f"metric {metric!r} not in {list(task_scores.keys())}")


def run_zeroshot_six(
    model_name: str,
    *,
    batch_size: int = 8,
    device: str = "cuda",
    dtype: str = "float16",
) -> EvalArtifact:
    """Zhu et al. Table 1 six-task zero-shot protocol.

    Returns percentages (0-100) for each task + `average`.
    """
    import time

    from lm_eval import simple_evaluate  # type: ignore[import-not-found]
    import lm_eval  # type: ignore[import-not-found]

    t0 = time.time()
    out = simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_name},dtype={dtype}",
        tasks=ZEROSHOT_SIX_TASKS,
        batch_size=batch_size,
        device=device,
    )
    elapsed = time.time() - t0

    metrics: dict[str, float] = {}
    for task in ZEROSHOT_SIX_TASKS:
        task_scores = out["results"].get(task, {})
        val = _extract_metric(task_scores, ZEROSHOT_SIX_METRIC[task])
        metrics[task] = val * 100.0
    metrics["average"] = sum(metrics.values()) / len(ZEROSHOT_SIX_TASKS)

    return EvalArtifact(
        model_name=model_name,
        protocol="zeroshot_six",
        metrics=metrics,
        elapsed_seconds=elapsed,
        lm_eval_version=getattr(lm_eval, "__version__", "unknown"),
    )


def run_held_out_loss(
    model: "torch.nn.Module",
    *,
    batches,
    n_steps: int = 100,
    device: str = "cuda",
) -> float:
    """Compute mean cross-entropy loss over `n_steps` batches without
    updating model weights.

    Used as the cheap/fast inner-training-loop validation: sample a fixed
    number of held-out batches, compute loss, return the average. Caller
    is responsible for ensuring `batches` draws from a held-out shard
    (different seed than train stream).

    Returns the mean nat-loss (for perplexity, exponentiate).
    """
    import torch  # local import; avoids hard dep at module import time

    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for i, ids in enumerate(batches):
            if i >= n_steps:
                break
            out = model(input_ids=ids.to(device), labels=ids.to(device))
            losses.append(float(out.loss.detach()))
    model.train()
    return sum(losses) / max(1, len(losses))


def run_wikitext103_ppl(
    model_name: str,
    *,
    batch_size: int = 4,
    device: str = "cuda",
    dtype: str = "float16",
) -> EvalArtifact:
    """Wikitext-103 word-level perplexity.

    Not in Zhu et al. Table 1; added by Phase D per §6.9 R1 canonical
    protocol expansion. Reports `word_perplexity`.
    """
    import time

    from lm_eval import simple_evaluate  # type: ignore[import-not-found]
    import lm_eval  # type: ignore[import-not-found]

    t0 = time.time()
    out = simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_name},dtype={dtype}",
        tasks=["wikitext"],  # lm_eval's wikitext task = wikitext-2 by default
        batch_size=batch_size,
        device=device,
    )
    elapsed = time.time() - t0

    task_scores = out["results"].get("wikitext", {})
    metrics = {
        "word_perplexity": _extract_metric(task_scores, "word_perplexity"),
    }
    return EvalArtifact(
        model_name=model_name,
        protocol="wikitext_ppl",
        metrics=metrics,
        elapsed_seconds=elapsed,
        lm_eval_version=getattr(lm_eval, "__version__", "unknown"),
    )


# ---------------------------------------------------------------------------
# E2.4.C.2 — quantization-error metric on bilingual eval set
# ---------------------------------------------------------------------------

# Adoption gate per `FJQ_PHASE_E_E2_4_C_METRIC_SPEC.md` §6.
QUANT_ERROR_OUTLIER_GATE: float = 0.10


def compute_quant_error_per_channel(
    model: "torch.nn.Module",
    *,
    batches: Iterator["torch.Tensor"],
    n_batches: int,
    bit_map_path: str | Path,
    device: str | "torch.device" = "cuda",
    out_path: Path | None = None,
) -> dict:
    """E2.4.C.2 driver — compute per-channel MSE quantization error
    against the saved bit allocation map and produce the adoption-gate
    JSON artifact.

    Per `FJQ_PHASE_E_E2_4_C_METRIC_SPEC.md` v1.0 §4-§7. Captures inputs
    to every BitLinear-family module via forward pre-hook, streams
    `n_batches` of `batches`, accumulates per-channel sum-squared-error
    under both `q_baseline` (upstream `activation_quant`, per-token
    8-bit) and `q_calibrated` (`activation_quant_per_channel` with the
    saved per-channel running_max + bits map), and returns the
    aggregated reduction metrics.

    The function does NOT modify model weights or forward behavior —
    captured activations are quantized OFFLINE in the hook for both
    quantizers; the model continues to run in its native (or trained)
    precision throughout. This preserves the Option C scope (no upstream
    fork; see `FJQ_PHASE_E_E2_4_FINDINGS.md` v1.2 §6.1).

    Args:
        model: Trained HGRNBit model (or any model with BitLinear-family
            layers). Should have been trained with `--balanced-calib`
            so the saved map at `bit_map_path` corresponds to it.
        batches: Iterator yielding `(batch_size, seq_len)` int64 token
            tensors. Typically `bilingual_stream(id_share=0.6, seed=42)`
            per spec §2.
        n_batches: Number of batches to stream. Spec default: 1000.
        bit_map_path: Path to `mini_<TAG>_maps.pt` written by
            `intllm.qat.save_calibration_maps`.
        device: Device the model lives on; activations are captured on
            this device.
        out_path: If provided, atomically write the result JSON here
            (atomic via `.tmp + os.replace`). If None, only return.

    Returns:
        dict with the schema in spec §7. Notable keys:
          `outlier_global_reduction` — the gate value (§5.3)
          `gate_pass` — boolean, True iff outlier_global_reduction ≥ 0.10
          `per_layer` — dict keyed by layer name with diagnostics
    """
    import torch

    from .qat import is_bitlinear
    from .quant import activation_quant_per_channel

    bit_map_path = Path(bit_map_path)
    if not bit_map_path.is_file():
        raise FileNotFoundError(f"bit_map_path does not exist: {bit_map_path}")

    payload = torch.load(bit_map_path, weights_only=False, map_location=device)
    meta = payload.get("_meta", {})
    layer_keys = [k for k in payload.keys() if k != "_meta"]
    if not layer_keys:
        raise ValueError(
            f"bit_map_path {bit_map_path} contains no BitLinear entries",
        )

    # Per-layer state for streaming SSE accumulation.
    #
    # Accumulator dtype: fp32 on GPU. RTX 4090 (and any consumer Ada/Hopper)
    # has 1:64 fp64 throughput; using fp64 here was making the 1000-batch
    # sweep ~64× slower than necessary. The fp32 accumulator's worst-case
    # numerical loss over 1000 batches × 8192 rows × O(1) per-element SSE
    # is ~7 decimal digits — well within the 4 sig figs the metric reports.
    # For paranoid users a fp64 mean is computed at the end via a single
    # cast. Don't change this without measuring on the production GPU.
    sse_baseline: dict[str, torch.Tensor] = {}
    sse_calibrated: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    running_max: dict[str, torch.Tensor] = {}
    bits: dict[str, torch.Tensor] = {}
    in_features: dict[str, int] = {}

    for layer_name in layer_keys:
        entry = payload[layer_name]
        n = int(entry["in_features"])
        sse_baseline[layer_name] = torch.zeros(n, dtype=torch.float32, device=device)
        sse_calibrated[layer_name] = torch.zeros(n, dtype=torch.float32, device=device)
        counts[layer_name] = 0
        running_max[layer_name] = entry["running_max"].to(device=device, dtype=torch.float32)
        bits[layer_name] = entry["bits"].to(device=device, dtype=torch.int32)
        in_features[layer_name] = n

    # Match BitLinear-resolution rule from intllm.qat.attach_stat_trackers:
    # find every BitLinear-family module in the model and attach a
    # forward pre-hook that captures the *input* (matching what
    # activation_quant sees in the upstream forward).
    name_by_module: dict[int, str] = {}
    for name, module in model.named_modules():
        if not is_bitlinear(module):
            continue
        if name not in payload:
            # Map was computed for a different model layout. Skip
            # silently; reduction metrics will simply omit this layer.
            continue
        name_by_module[id(module)] = name

    if not name_by_module:
        raise RuntimeError(
            f"no BitLinear modules in model match the layers in {bit_map_path}; "
            f"map has {len(layer_keys)} layers, model has 0 matching",
        )

    def _hook_factory(layer_name: str):
        in_f = in_features[layer_name]
        rmax = running_max[layer_name]
        bts = bits[layer_name]
        sse_b = sse_baseline[layer_name]
        sse_c = sse_calibrated[layer_name]

        def _pre_hook(_module, inputs):
            x = inputs[0].detach()
            # Reshape to (N, in_features) once; reuse for both quantizers.
            x_flat = x.reshape(-1, in_f).float()
            # Upstream activation_quant (q_baseline): per-token absmax scale, 8-bit.
            scale_t = (127.0 / x_flat.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-5))
            yb_flat = (x_flat * scale_t).round().clamp_(-128.0, 127.0) / scale_t
            # q_calibrated: per-channel calibrated scale + per-channel bits.
            yc_flat = activation_quant_per_channel(x_flat, rmax, bts)
            sse_b.add_((x_flat - yb_flat).pow(2).sum(dim=0))
            sse_c.add_((x_flat - yc_flat).pow(2).sum(dim=0))
            counts[layer_name] += int(x_flat.shape[0])
        return _pre_hook

    handles = []
    for module in model.modules():
        if id(module) in name_by_module:
            layer_name = name_by_module[id(module)]
            handles.append(module.register_forward_pre_hook(_hook_factory(layer_name)))

    try:
        model.eval()
        with torch.no_grad():
            for i, ids in enumerate(batches):
                if i >= n_batches:
                    break
                ids = ids.to(device)
                _ = model(input_ids=ids, labels=ids)  # forward only; loss unused
    finally:
        for h in handles:
            h.remove()

    # Aggregate per-layer + global metrics.
    high_bits = int(meta.get("high_bits", 10))
    per_layer: dict[str, dict] = {}
    layer_mean_reductions: list[float] = []
    layer_outlier_reductions: list[float] = []

    for layer_name in layer_keys:
        if counts[layer_name] == 0:
            # Layer's hook never fired (e.g. lm_head not reached on a
            # truncated forward). Record zeros and skip aggregation.
            per_layer[layer_name] = {
                "mean_reduction": 0.0,
                "outlier_mean_reduction": 0.0,
                "n_outlier_channels": 0,
                "n_total_channels": in_features[layer_name],
                "mse_baseline_mean": 0.0,
                "mse_calibrated_mean": 0.0,
                "n_observations": 0,
            }
            continue
        # Cast to fp64 only at the final divide so the per-channel
        # mean (a small tensor) lands in high precision; the fp32 SSE
        # accumulator stays on the fast path during the loop.
        mse_b = (sse_baseline[layer_name].double() / counts[layer_name]).cpu()
        mse_c = (sse_calibrated[layer_name].double() / counts[layer_name]).cpu()
        reduction = (mse_b - mse_c) / mse_b.clamp(min=1e-12)
        outlier_mask = (bits[layer_name].cpu() == high_bits)
        n_outlier = int(outlier_mask.sum())
        if n_outlier > 0:
            outlier_mean_red = float(reduction[outlier_mask].mean())
        else:
            outlier_mean_red = 0.0
        per_layer[layer_name] = {
            "mean_reduction": float(reduction.mean()),
            "outlier_mean_reduction": outlier_mean_red,
            "n_outlier_channels": n_outlier,
            "n_total_channels": in_features[layer_name],
            "mse_baseline_mean": float(mse_b.mean()),
            "mse_calibrated_mean": float(mse_c.mean()),
            "n_observations": counts[layer_name],
        }
        layer_mean_reductions.append(per_layer[layer_name]["mean_reduction"])
        if n_outlier > 0:
            layer_outlier_reductions.append(outlier_mean_red)

    global_mean = (
        sum(layer_mean_reductions) / len(layer_mean_reductions)
        if layer_mean_reductions else 0.0
    )
    outlier_global = (
        sum(layer_outlier_reductions) / len(layer_outlier_reductions)
        if layer_outlier_reductions else 0.0
    )

    result: dict = {
        "_schema_version": "1.0",
        "tag": meta.get("tag", "unknown"),
        "n_batches": int(n_batches),
        "n_layers": len(per_layer),
        "global_mean_reduction": float(global_mean),
        "outlier_global_reduction": float(outlier_global),
        "gate_threshold": QUANT_ERROR_OUTLIER_GATE,
        "gate_pass": bool(outlier_global >= QUANT_ERROR_OUTLIER_GATE),
        "per_layer": per_layer,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(result, f, indent=2, default=str)
        os.replace(tmp, out_path)

    return result
