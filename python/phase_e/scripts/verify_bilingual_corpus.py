#!/usr/bin/env python3
"""
Phase E E1.6 — Bilingual corpus packaging verification gate.

Per CLAUDE §6.8 R3 (prevention layer per phase): wired to
`make verify-bilingual-corpus`. Asserts the spec doc
`docs/FJQ_PHASE_E_BILINGUAL_CORPUS_V1.md` stays consistent with the
on-disk evidence: ID manifests + EN config shim + tokenizer measurement
file.

What it covers (must stay green for E1.6 gate):

  1. Both ID manifests exist + parseable
  2. ID `bytes_text_kept` aggregate within ±0.5 GB of spec §1.3 (38.92 GB)
  3. ID `docs_kept` aggregate matches spec §1.3 exactly (10,662,911)
  4. ID `dedup_rate` within ±0.005% of spec §1.3 (0.0254%)
  5. EN cap from `intllm_en.bilingual_mix_summary()` matches spec §1.2
     (10.27 B ± 0.05 B at 60:40 default)
  6. EN repo selected = SLIMPAJAMA_STRETCH_REPO at 60:40 default
  7. synthetic_mix_ratio == 0.0 (E1.5 deferred per plan v1.7/v1.8)
  8. License attribution covers all sources actually used (no missing rows)

Runtime: < 1 s (no network, no torch).
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PHASE_E = REPO_ROOT / "python" / "phase_e"
SPEC_DOC = REPO_ROOT / "docs" / "FJQ_PHASE_E_BILINGUAL_CORPUS_V1.md"
MANIFEST_DIR = REPO_ROOT / "data" / "phase_e" / "corpus_id_dedup_exact"

# Spec table §1.3 reference values — must stay in sync with the doc.
SPEC_ID_DOCS_KEPT = 10_662_911
SPEC_ID_BYTES_KEPT = 38.92e9  # 38.92 GB
SPEC_ID_DEDUP_RATE = 0.000254  # 0.0254%
SPEC_EN_CAP_AT_60_40 = 10.27e9  # 10.27 B
SPEC_TOTAL_TOKENS_AT_60_40 = 25.67e9  # 25.67 B

# Tolerances (per spec §5.3).
BYTES_TOL = 0.5e9  # ±0.5 GB
DEDUP_TOL = 0.00005  # ±0.005% absolute
EN_CAP_TOL = 0.05e9  # ±0.05 B

# Sources actually used in v1.0 corpus (must match spec §2 attribution).
EXPECTED_SOURCES = {
    "wikimedia__wikipedia",
    "HuggingFaceFW__fineweb-2",
}
# Required license-attribution names appearing in spec §2.
EXPECTED_LICENSE_NAMES = {"CC BY-SA 4.0", "ODC-By 1.0", "Apache 2.0"}


def main() -> int:
    sys.path.insert(0, str(PHASE_E))
    intllm_en = importlib.import_module("intllm_en")

    # 1. Both ID manifests exist + parseable ──────────────────────────
    manifests: dict[str, dict] = {}
    for src in EXPECTED_SOURCES:
        m_path = MANIFEST_DIR / src / "_manifest.json"
        assert m_path.is_file(), f"manifest missing: {m_path}"
        manifests[src] = json.loads(m_path.read_text(encoding="utf-8"))

    # 2. ID bytes_text_kept aggregate ──────────────────────────────────
    total_bytes = sum(m["bytes_text_kept"] for m in manifests.values())
    assert abs(total_bytes - SPEC_ID_BYTES_KEPT) <= BYTES_TOL, (
        f"ID bytes_text_kept drift: {total_bytes/1e9:.3f} GB vs spec "
        f"{SPEC_ID_BYTES_KEPT/1e9:.3f} GB (tol ±{BYTES_TOL/1e9:.3f} GB)"
    )

    # 3. ID docs_kept aggregate exact ───────────────────────────────────
    total_docs = sum(m["docs_kept"] for m in manifests.values())
    assert total_docs == SPEC_ID_DOCS_KEPT, (
        f"ID docs_kept exact mismatch: {total_docs} vs spec {SPEC_ID_DOCS_KEPT}"
    )

    # 4. ID dedup_rate aggregate ─────────────────────────────────────────
    total_in = sum(m["docs_in"] for m in manifests.values())
    total_dropped = sum(m["docs_dropped"] for m in manifests.values())
    aggregate_rate = total_dropped / total_in if total_in else 0.0
    assert abs(aggregate_rate - SPEC_ID_DEDUP_RATE) <= DEDUP_TOL, (
        f"ID dedup_rate drift: {aggregate_rate*100:.4f}% vs spec "
        f"{SPEC_ID_DEDUP_RATE*100:.4f}% (tol ±{DEDUP_TOL*100:.4f}%)"
    )

    # 5. EN cap from shim matches spec ─────────────────────────────────
    summary = intllm_en.bilingual_mix_summary()
    assert (
        abs(summary["en_tokens"] - SPEC_EN_CAP_AT_60_40) <= EN_CAP_TOL
    ), (
        f"EN cap drift: {summary['en_tokens']/1e9:.3f} B vs spec "
        f"{SPEC_EN_CAP_AT_60_40/1e9:.3f} B (tol ±{EN_CAP_TOL/1e9:.3f} B)"
    )
    assert summary["ratio_id_share"] == 0.6, (
        f"default ratio drift: {summary['ratio_id_share']} vs 0.6"
    )

    # 6. EN repo selected = STRETCH at default ratio ────────────────────
    assert summary["slimpajama_repo"] == intllm_en.SLIMPAJAMA_STRETCH_REPO, (
        f"unexpected EN repo at 60:40: {summary['slimpajama_repo']}; "
        f"expected {intllm_en.SLIMPAJAMA_STRETCH_REPO} (Stretch — EN cap "
        f"exceeds 6 B threshold)"
    )

    # 7. synthetic_mix_ratio == 0.0 ─────────────────────────────────────
    # Currently a static spec assertion (§4 of the doc says 0%). The
    # synthetic-mix tracking infrastructure (verify-synthetic-mix Makefile
    # target from plan §3 E1.5) doesn't exist yet because E1.5 is deferred.
    # When E1.5 reactivates in Phase F, this assertion becomes a live
    # check against an actual mix manifest.
    spec_text = SPEC_DOC.read_text(encoding="utf-8")
    assert (
        "**Synthetic-mix-ratio**" not in spec_text
        or "synthetic_mix_ratio == 0.0" in spec_text
        or "synthetic_mix_ratio` == 0.0" in spec_text
        or "**0%**" in spec_text
    ), "spec §4 must declare synthetic mix 0% for v1.0"
    # Direct match on the §1.3 row's "Synthetic-mix-ratio" cell:
    assert re.search(
        r"\|\s*Synthetic-mix-ratio\s*\|\s*\*\*0%\*\*\s*\|", spec_text
    ), "spec §1.3 'Synthetic-mix-ratio' must equal **0%** for v1.0"

    # 8. License attribution covers all sources actually used ──────────
    # Look for each source name in the spec's §2 attribution table.
    spec_section_2 = re.search(
        r"## 2\. License attribution.*?(?=## 3\.)", spec_text, re.DOTALL
    )
    assert spec_section_2 is not None, "spec §2 License attribution missing"
    section_2_body = spec_section_2.group(0)
    for src in EXPECTED_SOURCES:
        # The src string is e.g. "wikimedia__wikipedia"; the spec uses
        # human names like "Wikipedia ID" and "FineWeb-2 ID". Probe for
        # the relevant friendly names + a license string.
        friendly = src.replace("__", "/").replace("_", "-")  # rough
        if "wikipedia" in src.lower():
            assert "Wikipedia" in section_2_body, "spec §2 missing Wikipedia row"
            assert "CC BY-SA 4.0" in section_2_body, "spec §2 missing CC BY-SA"
        elif "fineweb" in src.lower():
            assert "FineWeb-2" in section_2_body, "spec §2 missing FineWeb-2 row"
            assert "ODC-By" in section_2_body, "spec §2 missing ODC-By"
    # SlimPajama (EN streaming) attribution check too — we use it via shim
    # even though no parquet on disk.
    assert "SlimPajama" in section_2_body, "spec §2 missing SlimPajama row"
    assert "Apache 2.0" in section_2_body, "spec §2 missing Apache 2.0 (SlimPajama)"

    # All checks passed.
    print("verify-bilingual-corpus: 8/8 invariants PASS")
    print(
        f"  ID:  {total_docs:>11,} docs / {total_bytes/1e9:6.3f} GB / "
        f"~{total_bytes/1e9/intllm_en.ID_BYTES_PER_TOKEN:.3f} B tokens "
        f"({aggregate_rate*100:.4f}% dedup)"
    )
    print(
        f"  EN:  cap {summary['en_tokens']/1e9:6.3f} B at "
        f"{summary['ratio_id_share']:.0%}:{1-summary['ratio_id_share']:.0%} "
        f"→ {summary['slimpajama_repo']}"
    )
    print(
        f"  Mix: total {(summary['id_tokens']+summary['en_tokens'])/1e9:.3f} B "
        f"= {summary['id_tokens']/1e9:.3f} ID + {summary['en_tokens']/1e9:.3f} EN  "
        f"(synthetic 0%)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
