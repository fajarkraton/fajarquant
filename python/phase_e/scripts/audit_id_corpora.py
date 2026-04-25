#!/usr/bin/env python3
"""
Phase E E0.0.1 — Inventory of Indonesian-language pretraining corpora.

Per FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md §3 E0.0.1 + §E1.1:
- Probe known sources for availability, license, approximate token count
- Distinguish HF-Datasets-available vs manual-acquisition sources
- Output: stdout summary + JSON report + exit code

Abort threshold per plan §2 Q1: <8B usable tokens post-dedup → Phase E pivot.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PHASE_E_RESULTS_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "paper"
    / "intllm"
    / "results"
    / "phase_e"
)
DEFAULT_OUTPUT = PHASE_E_RESULTS_DIR / "e0_id_corpora_inventory.json"

# Token-count tiers from plan §E1.1 — refined empirically as actual probes complete.
TIER_HIGH = "high"
TIER_MEDIUM = "medium"
TIER_LOW = "low"


@dataclass
class CorpusSource:
    name: str
    hf_dataset_id: str | None
    raw_url: str | None
    license_known: str
    license_commercial_ok: bool | None
    quality_tier: str
    plan_estimate_tokens_b_low: float
    plan_estimate_tokens_b_high: float
    notes: str
    status: str = "unprobed"
    probed_at: str | None = None
    probe_error: str | None = None
    measured_size_bytes: int | None = None
    measured_num_examples: int | None = None


PLAN_E1_1_SOURCES: list[CorpusSource] = [
    CorpusSource(
        name="CC-100 Indonesian",
        hf_dataset_id="cc100",
        raw_url="https://data.statmt.org/cc-100/",
        license_known="custom (CC-100 terms)",
        license_commercial_ok=False,
        quality_tier=TIER_MEDIUM,
        plan_estimate_tokens_b_low=10.0,
        plan_estimate_tokens_b_high=15.0,
        notes="needs heavy dedup; lang=id config",
    ),
    CorpusSource(
        name="OSCAR Indonesian v23.01",
        hf_dataset_id="oscar-corpus/OSCAR-2301",
        raw_url=None,
        license_known="cc0-1.0",
        license_commercial_ok=True,
        quality_tier=TIER_MEDIUM,
        plan_estimate_tokens_b_low=5.0,
        plan_estimate_tokens_b_high=8.0,
        notes="language-detected, partial dedup; gated dataset (HF auth required)",
    ),
    CorpusSource(
        name="Wikipedia Indonesian",
        hf_dataset_id="wikimedia/wikipedia",
        raw_url="https://dumps.wikimedia.org/idwiki/",
        license_known="CC-BY-SA 4.0",
        license_commercial_ok=True,
        quality_tier=TIER_HIGH,
        plan_estimate_tokens_b_low=0.5,
        plan_estimate_tokens_b_high=1.0,
        notes="curated; ~700K articles; 20231101.id config",
    ),
    CorpusSource(
        name="Indonesian news crawls (Detik/Kompas/Tempo aggregations)",
        hf_dataset_id=None,
        raw_url="manual",
        license_known="per-source-varies",
        license_commercial_ok=None,
        quality_tier=TIER_HIGH,
        plan_estimate_tokens_b_low=3.0,
        plan_estimate_tokens_b_high=5.0,
        notes="needs licensing review per E0.0.7; check uonlp/CulturaX or HuggingFaceFW/fineweb-ind subset",
    ),
    CorpusSource(
        name="Indonesian books OCR (gutenberg-id, archive.org)",
        hf_dataset_id=None,
        raw_url="https://archive.org/details/openlibrary",
        license_known="public-domain",
        license_commercial_ok=True,
        quality_tier=TIER_HIGH,
        plan_estimate_tokens_b_low=0.5,
        plan_estimate_tokens_b_high=1.0,
        notes="OCR quality variable; manual curation needed",
    ),
    CorpusSource(
        name="Indonesian academic / research (theses, arXiv-id)",
        hf_dataset_id=None,
        raw_url="https://repository.ugm.ac.id/  + ITB + UI repos",
        license_known="varies (mostly academic-fair-use)",
        license_commercial_ok=None,
        quality_tier=TIER_HIGH,
        plan_estimate_tokens_b_low=0.2,
        plan_estimate_tokens_b_high=0.5,
        notes="small but high signal; per-institution scraping consent required",
    ),
    CorpusSource(
        name="CulturaX Indonesian",
        hf_dataset_id="uonlp/CulturaX",
        raw_url=None,
        license_known="ODC-By + per-source-varies",
        license_commercial_ok=True,
        quality_tier=TIER_HIGH,
        plan_estimate_tokens_b_low=8.0,
        plan_estimate_tokens_b_high=20.0,
        notes="NEW v1.1: heavily-cleaned multilingual corpus; lang=id subset; superior to raw CC-100 + OSCAR",
    ),
    CorpusSource(
        name="MADLAD-400 Indonesian (Google)",
        hf_dataset_id="allenai/MADLAD-400",
        raw_url=None,
        license_known="ODC-By",
        license_commercial_ok=True,
        quality_tier=TIER_MEDIUM,
        plan_estimate_tokens_b_low=5.0,
        plan_estimate_tokens_b_high=10.0,
        notes="NEW v1.1: 419 languages including Indonesian; cleaned web crawl",
    ),
    CorpusSource(
        name="FineWeb-2 Indonesian (HuggingFaceFW)",
        hf_dataset_id="HuggingFaceFW/fineweb-2",
        raw_url=None,
        license_known="ODC-By",
        license_commercial_ok=True,
        quality_tier=TIER_HIGH,
        plan_estimate_tokens_b_low=10.0,
        plan_estimate_tokens_b_high=30.0,
        notes="NEW v1.1: latest high-quality multilingual web; ind_Latn subset",
    ),
]


def probe_hf_dataset(source: CorpusSource, timeout_s: float = 15.0) -> CorpusSource:
    """Query HF Datasets API for size + example count without downloading."""
    if source.hf_dataset_id is None:
        source.status = "manual-required"
        return source
    try:
        from huggingface_hub import HfApi
    except ImportError:
        source.status = "probe-skipped"
        source.probe_error = "huggingface_hub not installed"
        return source

    api = HfApi()
    started = time.monotonic()
    try:
        info = api.dataset_info(source.hf_dataset_id, timeout=timeout_s)
        source.status = "available"
        source.probed_at = datetime.now(timezone.utc).isoformat()
        if hasattr(info, "size_categories") and info.size_categories:
            source.notes += f" | size_categories={info.size_categories}"
        if hasattr(info, "tags") and info.tags:
            ind_tags = [t for t in info.tags if "ind" in t.lower() or "id" in t.lower()][:5]
            if ind_tags:
                source.notes += f" | tags={ind_tags}"
    except Exception as e:
        source.status = "probe-failed"
        source.probe_error = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        elapsed = time.monotonic() - started
        if elapsed > 5.0:
            source.notes += f" | probe_elapsed_s={elapsed:.1f}"

    return source


def render_summary(sources: list[CorpusSource]) -> str:
    lines = []
    lines.append("=" * 110)
    lines.append(f"{'SOURCE':<48} {'STATUS':<16} {'EST_TOKENS_B':>14} {'TIER':>8} {'COMMERCIAL':>12}")
    lines.append("-" * 110)
    total_low = total_high = 0.0
    for s in sources:
        est = f"{s.plan_estimate_tokens_b_low:.1f}-{s.plan_estimate_tokens_b_high:.1f}"
        commercial = (
            "yes" if s.license_commercial_ok is True
            else "no" if s.license_commercial_ok is False
            else "review"
        )
        lines.append(
            f"{s.name[:47]:<48} {s.status:<16} {est:>14} {s.quality_tier:>8} {commercial:>12}"
        )
        total_low += s.plan_estimate_tokens_b_low
        total_high += s.plan_estimate_tokens_b_high
    lines.append("-" * 110)
    lines.append(
        f"{'TOTAL (raw plan estimate)':<48} {'':<16} {total_low:>5.1f}-{total_high:.1f} B {'':>8} {'':>12}"
    )
    lines.append("")
    lines.append("Plan §2 Q1 abort threshold: < 8.0 B usable tokens post-dedup → Phase E pivot.")
    expected_dedup_rate = 0.6
    expected_post_dedup_low = total_low * expected_dedup_rate
    expected_post_dedup_high = total_high * expected_dedup_rate
    lines.append(
        f"Expected post-dedup (assuming {expected_dedup_rate:.0%} retention): "
        f"{expected_post_dedup_low:.1f}-{expected_post_dedup_high:.1f} B"
    )
    if expected_post_dedup_low >= 8.0:
        lines.append("→ ABOVE abort threshold even at low-end estimate; PROCEED to E1.")
    elif expected_post_dedup_high >= 8.0:
        lines.append("→ Above threshold ONLY at high-end; CAUTION — empirical token-count needed in E1.")
    else:
        lines.append("→ BELOW threshold; review pivot path before E1 commitment.")
    lines.append("=" * 110)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any high-tier source is unreachable or below abort threshold",
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip HF API probes (offline mode); use plan estimates only",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    sources = list(PLAN_E1_1_SOURCES)

    if not args.no_probe:
        for source in sources:
            probe_hf_dataset(source)

    print(render_summary(sources))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "_phase": "FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md §3 E0.0.1",
        "_plan_ref": "§E1.1 Indonesian corpus assembly",
        "_abort_threshold_b_post_dedup": 8.0,
        "_generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": [asdict(s) for s in sources],
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport written: {args.output}")

    if args.strict:
        unreachable_high = [
            s for s in sources
            if s.quality_tier == TIER_HIGH and s.status not in ("available", "manual-required")
        ]
        if unreachable_high:
            print(f"\nSTRICT MODE: {len(unreachable_high)} high-tier source(s) unreachable:")
            for s in unreachable_high:
                print(f"  - {s.name}: {s.probe_error or s.status}")
            return 1
        total_low_b = sum(s.plan_estimate_tokens_b_low for s in sources) * 0.6
        if total_low_b < 8.0:
            print(f"\nSTRICT MODE: post-dedup low estimate {total_low_b:.1f}B < 8.0B abort threshold")
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
