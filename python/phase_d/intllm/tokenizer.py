"""IntLLM tokenizer — thin wrapper over Mistral v3 (32K vocab).

Decision committed in `FJQ_PHASE_D_P3_FINDINGS.md` (C.P3.1 gate).
Mistral v3 chosen over SmolLM 49K + custom BPE because it gives direct
comparability with Zhu et al.'s MatMul-Free LLM Table 1 numbers (which
uses Mistral tokenizer at 32K vocab).

Source: `mistralai/Mistral-7B-v0.3` on Hugging Face — Apache-2.0,
publicly downloadable (`tokenizer.model.v3` file), no authentication
required.

Special tokens:
  BOS = `<s>` (id 1)
  EOS = `</s>` (id 2)
  PAD = none by default — we set `pad_token = eos_token` for batched
        training, mirroring the standard Llama-family convention.
"""

from __future__ import annotations

from functools import lru_cache

from transformers import AutoTokenizer, PreTrainedTokenizerBase

DEFAULT_TOKENIZER_NAME = "mistralai/Mistral-7B-v0.3"
DEFAULT_VOCAB_SIZE = 32768  # Mistral v3 spec — fixed, not auto-detected


@lru_cache(maxsize=2)
def get_tokenizer(name: str = DEFAULT_TOKENIZER_NAME) -> PreTrainedTokenizerBase:
    """Return a cached HF tokenizer instance for `name`.

    Sets `pad_token = eos_token` because Mistral v3 has no PAD token by
    default, and the standard Llama-family training convention is to
    re-use EOS for padding.

    Cached via `lru_cache` so repeated calls within a process don't
    re-download or re-build the tokenizer.
    """
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def encode_batch(
    texts: list[str],
    *,
    seq_len: int,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> list[list[int]]:
    """Tokenize + truncate a batch of strings to fixed `seq_len`.

    Returns a list of lists (no tensorisation) so callers downstream
    can choose torch / numpy / json. For training-ready batches use
    `slimpajama_stream` in `intllm.data` which packs tokens into
    `(batch_size, seq_len)` int64 tensors.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    out: list[list[int]] = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True)
        out.append(ids[:seq_len])
    return out


__all__ = [
    "DEFAULT_TOKENIZER_NAME",
    "DEFAULT_VOCAB_SIZE",
    "encode_batch",
    "get_tokenizer",
]
