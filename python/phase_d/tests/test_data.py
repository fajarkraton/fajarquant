"""Tests for intllm.data.

C.P6.4 (Track B step 4): interruption-safe streaming. Covers the
`_retry_iter` helper + HF timeout env-var defaults. Does NOT exercise
the real HuggingFace hub (that's integration scope + needs network).
"""
from __future__ import annotations

import os

import pytest


def test_hf_download_timeout_env_vars_set_on_import() -> None:
    """Importing intllm.data sets HF timeout env vars (unless user already did)."""
    # Sanity import — will raise if the module has a syntax error.
    import intllm.data  # noqa: F401

    assert os.environ.get("HF_DATASETS_DOWNLOAD_TIMEOUT") == "60", (
        "HF_DATASETS_DOWNLOAD_TIMEOUT should default to 60 after intllm.data import"
    )
    assert os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT") == "60"


def test_retry_iter_passes_through_on_success() -> None:
    """Factory yields 3 items without exceptions → retry_iter yields all 3."""
    from intllm.data import _retry_iter

    calls = {"n": 0}

    def factory(attempt: int):
        calls["n"] += 1
        assert attempt == 0
        yield from [1, 2, 3]

    out = list(_retry_iter(factory, on_retry=lambda a, e: None))
    assert out == [1, 2, 3]
    assert calls["n"] == 1  # no retries


def test_retry_iter_recovers_from_transient_error() -> None:
    """Factory fails attempt 0 with OSError (retryable), succeeds attempt 1."""
    from intllm.data import _retry_iter

    attempts_seen: list[int] = []

    def factory(attempt: int):
        attempts_seen.append(attempt)
        if attempt == 0:
            # Must raise from within a generator → use a helper func.
            def _explode():
                raise OSError("simulated CDN drop")
                yield  # pragma: no cover - unreachable
            yield from _explode()
        else:
            yield from ["a", "b"]

    out = list(_retry_iter(factory, on_retry=lambda a, e: None))
    assert out == ["a", "b"]
    assert attempts_seen == [0, 1]


def test_retry_iter_exhausts_max_attempts() -> None:
    """Factory always fails → raises after max_attempts."""
    from intllm.data import _retry_iter

    def factory(attempt: int):
        def _explode():
            raise OSError(f"fail #{attempt}")
            yield  # pragma: no cover
        yield from _explode()

    with pytest.raises(OSError, match="fail #"):
        list(_retry_iter(
            factory, max_attempts=3, on_retry=lambda a, e: None,
        ))


def test_retry_iter_propagates_non_retryable_exception() -> None:
    """Non-retryable (ValueError, not in default set) → propagates on attempt 0."""
    from intllm.data import _retry_iter

    def factory(attempt: int):
        def _explode():
            raise ValueError("logic bug, not network")
            yield  # pragma: no cover
        yield from _explode()

    with pytest.raises(ValueError, match="logic bug"):
        list(_retry_iter(factory, max_attempts=5, on_retry=lambda a, e: None))


def test_retry_iter_on_retry_callback_receives_attempt_and_exception() -> None:
    """on_retry is called with (attempt_number, exception) before the retry."""
    from intllm.data import _retry_iter

    observed: list[tuple[int, str]] = []

    def factory(attempt: int):
        if attempt < 2:
            def _explode():
                raise OSError(f"blip {attempt}")
                yield  # pragma: no cover
            yield from _explode()
        else:
            yield from [42]

    def on_retry(attempt: int, exc: BaseException) -> None:
        observed.append((attempt, str(exc)))

    out = list(_retry_iter(factory, on_retry=on_retry))
    assert out == [42]
    # 2 failed attempts (0, 1) → on_retry called twice, with attempt=1 then 2
    assert observed == [(1, "blip 0"), (2, "blip 1")]


def test_retry_iter_custom_retryable_set_only_catches_listed() -> None:
    """Passing a custom retryable tuple restricts what gets retried."""
    from intllm.data import _retry_iter

    # Only catch KeyError (normally NOT retryable); OSError should propagate.
    def factory(attempt: int):
        def _explode():
            raise OSError("would normally retry")
            yield  # pragma: no cover
        yield from _explode()

    with pytest.raises(OSError):
        list(_retry_iter(
            factory, retryable=(KeyError,), on_retry=lambda a, e: None,
        ))
