"""Tests for fluxgen.generator helpers — filename generation, model dispatch."""
import re
from unittest.mock import patch

import pytest

from fluxgen.generator import (
    DEFAULT_MODEL,
    MODEL_DEFAULTS,
    SUPPORTED_MODELS,
    _timestamp_filename,
    generate_random_filename,
)


# ── generate_random_filename ────────────────────────────────────────────────────


def test_generate_random_filename_uses_wonderwords_when_available():
    """Happy path: wonderwords returns 3 hyphenated words + .png."""
    fn = generate_random_filename()
    # 3 short words joined by hyphens
    assert re.match(r"^[a-z]+(-[a-z]+){2}\.png$", fn), f"unexpected: {fn}"


def test_generate_random_filename_fallback_when_wonderwords_unavailable():
    """When wonderwords isn't installed, fall back to timestamp+suffix."""
    with patch("fluxgen.generator._random_word", None):
        fn = generate_random_filename()

    # ms timestamp (13 digits) + 4 hex chars + .png
    assert re.match(r"^generated-\d{13}-[0-9a-f]{4}\.png$", fn), f"unexpected: {fn}"


def test_generate_random_filename_fallback_when_random_words_raises():
    """If wonderwords.random_words raises, fall back to timestamp+suffix."""
    fake_rw = patch("fluxgen.generator._random_word").start()
    fake_rw.random_words.side_effect = RuntimeError("boom")
    try:
        fn = generate_random_filename()
    finally:
        patch.stopall()

    assert re.match(r"^generated-\d{13}-[0-9a-f]{4}\.png$", fn), f"unexpected: {fn}"


def test_generate_random_filename_no_collision_under_load():
    """100 consecutive fallback calls produce 100 unique filenames.

    The bug being fixed: `int(time.time())` had 1-second resolution, so
    two generations in the same second produced the same filename. With
    millisecond resolution + random suffix, collisions are essentially
    impossible.
    """
    with patch("fluxgen.generator._random_word", None):
        fns = [generate_random_filename() for _ in range(100)]

    assert len(set(fns)) == 100, f"only {len(set(fns))}/100 unique filenames"


def test_generate_random_filename_unique_for_same_millisecond():
    """Two calls in the same millisecond must still differ (via random suffix)."""
    with patch("fluxgen.generator._random_word", None), \
         patch("fluxgen.generator.time.time", return_value=1_700_000_000.123):
        fns = {generate_random_filename() for _ in range(20)}

    assert len(fns) == 20, "Same-millisecond calls collided"


# ── _timestamp_filename ────────────────────────────────────────────────────────


def test_timestamp_filename_format():
    """Format: generated-{ms}-{4hex}.png"""
    fn = _timestamp_filename()
    assert re.match(r"^generated-\d{13}-[0-9a-f]{4}\.png$", fn), f"unexpected: {fn}"


def test_timestamp_filename_uses_millisecond_timestamp():
    """Timestamp should be milliseconds (13 digits) not seconds (10)."""
    with patch("fluxgen.generator.time.time", return_value=1_700_000_000.123):
        fn = _timestamp_filename()

    # 1_700_000_000.123 * 1000 = 1_700_000_000_123 (13 digits)
    assert "1700000000123" in fn


def test_timestamp_filename_unique_across_calls():
    """Direct test of the helper: 50 calls produce 50 unique names."""
    fns = {_timestamp_filename() for _ in range(50)}
    assert len(fns) == 50


# ── Module-level constants (sanity) ───────────────────────────────────────────


def test_supported_models_are_non_empty_strings():
    assert len(SUPPORTED_MODELS) >= 1
    for m in SUPPORTED_MODELS:
        assert isinstance(m, str)
        assert m  # non-empty


def test_default_model_is_in_supported_models():
    assert DEFAULT_MODEL in SUPPORTED_MODELS


def test_model_defaults_keys_are_subset_of_supported_models():
    """Every model with defaults listed should be a supported model."""
    extra = set(MODEL_DEFAULTS.keys()) - set(SUPPORTED_MODELS)
    # 'flux2-klein-edit' is used by edit command and isn't in generation SUPPORTED_MODELS
    extra -= {"flux2-klein-edit"}
    assert not extra, f"MODEL_DEFAULTS has unknown keys: {extra}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
