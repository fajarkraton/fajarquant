"""Tests for `intllm.tokenizer` — verify Mistral v3 wrapper works as
spec'd in `FJQ_PHASE_D_P3_FINDINGS.md`.

These tests download the tokenizer from HF on first run (a few MB).
HF cache makes subsequent runs ~instant.
"""

from __future__ import annotations

import pytest

from intllm.tokenizer import (
    DEFAULT_TOKENIZER_NAME,
    DEFAULT_VOCAB_SIZE,
    encode_batch,
    get_tokenizer,
)


def test_default_tokenizer_is_mistral_v3() -> None:
    assert DEFAULT_TOKENIZER_NAME == "mistralai/Mistral-7B-v0.3"
    assert DEFAULT_VOCAB_SIZE == 32768


def test_get_tokenizer_loads_and_has_pad_token() -> None:
    tok = get_tokenizer()
    assert tok.vocab_size == 32768
    # We deliberately set pad_token = eos_token in get_tokenizer.
    assert tok.pad_token == tok.eos_token
    assert tok.bos_token == "<s>"
    assert tok.eos_token == "</s>"


def test_encode_decode_round_trip() -> None:
    tok = get_tokenizer()
    text = "Hello world from Phase D"
    ids = tok.encode(text, add_special_tokens=True)
    decoded = tok.decode(ids)
    # Mistral inserts a leading <s> + a leading space; both round-trip cleanly
    assert "Hello world from Phase D" in decoded
    # First token should always be BOS
    assert ids[0] == tok.bos_token_id


def test_encode_batch_truncates_to_seq_len() -> None:
    long_text = " ".join(["word"] * 200)  # ~200 tokens after BPE
    ids_lists = encode_batch([long_text, "short"], seq_len=32)
    assert len(ids_lists) == 2
    assert all(len(ids) <= 32 for ids in ids_lists)
    assert len(ids_lists[0]) == 32  # truncated long text
    assert len(ids_lists[1]) < 32  # short text fits without truncation


def test_get_tokenizer_caches_instance() -> None:
    # lru_cache returns the same object on the second call
    a = get_tokenizer()
    b = get_tokenizer()
    assert a is b
