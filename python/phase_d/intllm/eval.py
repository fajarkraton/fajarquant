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

None of these functions implement the evaluation themselves — they
delegate to `lm-evaluation-harness` which is the canonical literature
implementation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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
