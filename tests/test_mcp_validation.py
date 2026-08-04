"""Tests for fluxgen_mcp.validation."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from fluxgen_mcp.errors import (
    E_INPUT_TOO_HIGHRES,
    E_INPUT_TOO_LARGE,
    E_INVALID_INPUT_IMAGE,
    MCPError,
)
from fluxgen_mcp.validation import validate_edit_inputs, validate_init_image


def _make_png(path: Path, size: tuple[int, int]) -> Path:
    Image.new("RGB", size, (255, 0, 0)).save(path)
    return path


def test_validate_edit_inputs_accepts_small_png(tmp_path: Path):
    p = _make_png(tmp_path / "small.png", (100, 100))
    out = validate_edit_inputs([str(p)], max_bytes=10_000, max_dimension=200)
    assert out == [p.resolve()]


def test_validate_edit_inputs_rejects_oversized_file(tmp_path: Path):
    p = tmp_path / "big.png"
    p.write_bytes(b"x" * 10_001)
    with pytest.raises(MCPError) as exc:
        validate_edit_inputs([str(p)], max_bytes=10_000, max_dimension=2000)
    assert exc.value.code == E_INPUT_TOO_LARGE


def test_validate_edit_inputs_rejects_oversized_dimensions(tmp_path: Path):
    p = _make_png(tmp_path / "hires.png", (3000, 100))
    with pytest.raises(MCPError) as exc:
        validate_edit_inputs([str(p)], max_bytes=10_000_000, max_dimension=1080)
    assert exc.value.code == E_INPUT_TOO_HIGHRES


def test_validate_edit_inputs_rejects_corrupt_file(tmp_path: Path):
    p = tmp_path / "corrupt.png"
    p.write_bytes(b"not actually an image")
    from fluxgen.exceptions import InvalidImageError

    with pytest.raises(InvalidImageError):
        validate_edit_inputs([str(p)], max_bytes=10_000, max_dimension=2000)


def test_validate_edit_inputs_rejects_missing_file(tmp_path: Path):
    p = tmp_path / "missing.png"
    with pytest.raises(MCPError) as exc:
        validate_edit_inputs([str(p)], max_bytes=10_000, max_dimension=2000)
    assert exc.value.code == E_INVALID_INPUT_IMAGE


def test_validate_edit_inputs_empty_list():
    from fluxgen_mcp.errors import E_BAD_ARG

    with pytest.raises(MCPError) as exc:
        validate_edit_inputs([], max_bytes=10_000, max_dimension=2000)
    assert exc.value.code == E_BAD_ARG


def test_validate_init_image_wrapper(tmp_path: Path):
    p = _make_png(tmp_path / "small.png", (50, 50))
    out = validate_init_image(str(p), max_bytes=10_000, max_dimension=200)
    assert out == p.resolve()
