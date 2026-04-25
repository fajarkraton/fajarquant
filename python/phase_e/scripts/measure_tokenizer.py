#!/usr/bin/env python3
"""
Phase E E0.0.3 — Tokenizer compression measurement on Indonesian corpus.

Per FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md v1.2 §3 E0.0.3 + §2 Q3:
- Measure Mistral-v3 32K tokenizer compression (bytes-per-token) on Indonesian
- Compare against English baseline for context
- Decision rule: > 4.0 bytes/token on ID -> custom 24K tokenizer mandatory in E2

Sample source: streaming Wikipedia ID + EN (no full corpus download needed).
N articles per language is configurable; default 2000 (representative sample).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PHASE_E_RESULTS_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "paper"
    / "intllm"
    / "results"
    / "phase_e"
)
DEFAULT_OUTPUT = PHASE_E_RESULTS_DIR / "e0_tokenizer_compression.json"

# Decision threshold per plan v1.2 §2 Q3
THRESHOLD_BYTES_PER_TOKEN_TRIGGER_CUSTOM_24K = 4.0
THRESHOLD_EFFICIENT_REFERENCE = 3.5


def load_tokenizer():
    """Load Mistral-v3 32K via intllm.tokenizer (Phase D module)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "phase_d"))
    try:
        from intllm.tokenizer import DEFAULT_TOKENIZER_NAME, get_tokenizer
    except ImportError:
        from transformers import AutoTokenizer
        DEFAULT_TOKENIZER_NAME = "mistralai/Mistral-7B-v0.3"
        return AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_NAME), DEFAULT_TOKENIZER_NAME
    return get_tokenizer(DEFAULT_TOKENIZER_NAME), DEFAULT_TOKENIZER_NAME


def stream_wikipedia(lang_code: str, n_articles: int) -> tuple[str, int]:
    """Stream N articles from wikimedia/wikipedia, return concatenated text + count."""
    from datasets import load_dataset

    config = f"20231101.{lang_code}"
    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)

    chunks: list[str] = []
    count = 0
    for example in ds:
        text = example.get("text", "")
        if not text:
            continue
        chunks.append(text)
        count += 1
        if count >= n_articles:
            break
    return "\n\n".join(chunks), count


def measure(text: str, tokenizer) -> dict:
    """Compute tokenization stats."""
    text_bytes = len(text.encode("utf-8"))
    text_chars = len(text)
    text_words = len(text.split())

    started = time.monotonic()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    elapsed = time.monotonic() - started

    n_tokens = len(token_ids)
    return {
        "n_chars": text_chars,
        "n_bytes": text_bytes,
        "n_words": text_words,
        "n_tokens": n_tokens,
        "bytes_per_token": text_bytes / n_tokens if n_tokens else 0.0,
        "chars_per_token": text_chars / n_tokens if n_tokens else 0.0,
        "tokens_per_word": n_tokens / text_words if text_words else 0.0,
        "tokenize_elapsed_s": round(elapsed, 3),
        "throughput_kchar_per_s": round(text_chars / 1000 / elapsed, 1) if elapsed else 0,
    }


def render_summary(report: dict) -> str:
    lines = []
    lines.append("=" * 88)
    lines.append("Tokenizer compression — Phase E E0.0.3")
    lines.append(f"Tokenizer: {report['tokenizer_name']} (vocab≈32K)")
    lines.append(f"Sample: {report['n_articles']} articles per language from wikimedia/wikipedia")
    lines.append("=" * 88)

    for lang_label, lang_data in report["per_language"].items():
        m = lang_data["measurement"]
        lines.append(f"\n[{lang_label}] {lang_data['articles_count']} articles, {m['n_bytes']:,} bytes")
        lines.append(f"  bytes/token   = {m['bytes_per_token']:.3f}")
        lines.append(f"  chars/token   = {m['chars_per_token']:.3f}")
        lines.append(f"  tokens/word   = {m['tokens_per_word']:.3f}")
        lines.append(f"  total tokens  = {m['n_tokens']:,}")

    id_bpt = report["per_language"]["Indonesian"]["measurement"]["bytes_per_token"]
    en_bpt = report["per_language"]["English"]["measurement"]["bytes_per_token"]
    ratio = id_bpt / en_bpt if en_bpt else 0

    lines.append("")
    lines.append("=" * 88)
    lines.append(
        f"ID vs EN compression ratio: {ratio:.3f}× "
        f"(ID needs {(ratio - 1) * 100:+.1f}% more tokens than EN for same bytes)"
    )
    lines.append("")
    lines.append(f"Decision threshold (plan §2 Q3): bytes/token > {THRESHOLD_BYTES_PER_TOKEN_TRIGGER_CUSTOM_24K}")
    lines.append(f"Efficient reference: bytes/token ≤ {THRESHOLD_EFFICIENT_REFERENCE}")
    lines.append("")

    if id_bpt > THRESHOLD_BYTES_PER_TOKEN_TRIGGER_CUSTOM_24K:
        lines.append(f"→ DECISION: bytes/token = {id_bpt:.3f} EXCEEDS 4.0 threshold")
        lines.append("→ Custom 24K Indonesian-optimized tokenizer MANDATORY in E2")
        report["decision"] = "custom_24k_required"
    elif id_bpt <= THRESHOLD_EFFICIENT_REFERENCE:
        lines.append(f"→ DECISION: bytes/token = {id_bpt:.3f} ≤ 3.5 (efficient)")
        lines.append("→ Mistral-v3 32K is efficient enough; KEEP as primary tokenizer")
        report["decision"] = "keep_mistral_v3_32k"
    else:
        lines.append(f"→ DECISION: bytes/token = {id_bpt:.3f} between 3.5–4.0 (acceptable)")
        lines.append("→ Mistral-v3 32K acceptable; custom 24K optional in v2 if quality margin tight")
        report["decision"] = "keep_mistral_v3_32k_optional_v2_custom"

    lines.append("=" * 88)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if decision is custom_24k_required")
    args = parser.parse_args()

    print(f"[E0.0.3] Loading Mistral-v3 32K tokenizer...")
    tokenizer, tokenizer_name = load_tokenizer()
    print(f"  loaded: {tokenizer_name}")

    languages = [("Indonesian", "id"), ("English", "en")]
    per_language = {}

    for label, lang_code in languages:
        print(f"\n[E0.0.3] Streaming {args.articles} {label} Wikipedia articles...")
        try:
            text, count = stream_wikipedia(lang_code, args.articles)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            return 3

        print(f"  collected: {count} articles, {len(text):,} chars")
        print(f"[E0.0.3] Tokenizing...")
        m = measure(text, tokenizer)
        per_language[label] = {"articles_count": count, "measurement": m}
        print(f"  bytes/token = {m['bytes_per_token']:.3f}")

    report = {
        "_phase": "FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md §3 E0.0.3",
        "_plan_ref": "§2 Q3 tokenizer compression decision",
        "_threshold_custom_24k_trigger": THRESHOLD_BYTES_PER_TOKEN_TRIGGER_CUSTOM_24K,
        "_threshold_efficient": THRESHOLD_EFFICIENT_REFERENCE,
        "_generated_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer_name": tokenizer_name,
        "tokenizer_vocab_size": len(tokenizer),
        "n_articles": args.articles,
        "per_language": per_language,
        "decision": None,
    }

    print()
    print(render_summary(report))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport written: {args.output}")

    if args.strict and report["decision"] == "custom_24k_required":
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
