"""Tests for the edit_image tool wrapper.

The underlying ImageEditor is mocked at the seam so the tests
don't load diffusers / GGUF weights.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from fluxgen.exceptions import FluxgenError
from fluxgen_mcp.config import MCPSettings
from fluxgen_mcp.errors import (
    E_BAD_ARG,
    E_INPUT_TOO_LARGE,
    E_INVALID_INPUT_IMAGE,
    E_MODEL,
    MCPError,
)
from fluxgen_mcp.tools.edit import edit_image_tool


def _settings(tmp_path: Path, **overrides) -> MCPSettings:
    base = dict(
        output_root=str(tmp_path / "out"),
        max_width=1920,
        max_height=1920,
        max_steps=50,
        max_prompt_chars=2000,
        max_concurrent_jobs=1,
        max_queue_depth=4,
        per_call_timeout_s=600.0,
        allowed_generation_models=("zimage-turbo",),
        allowed_edit_models=("flux2-klein", "qwen-image-edit"),
        prompt_blocklist=(),
        audit_log_path=str(tmp_path / "audit.log"),
        pause_sentinel_path=str(tmp_path / "paused"),
        pid_file_path=str(tmp_path / "pid"),
        input_max_bytes=20_000_000,
        input_max_dimension=1080,
    )
    base.update(overrides)
    return MCPSettings(**base)


def _make_png(path: Path, size: tuple[int, int]) -> Path:
    Image.new("RGB", size, (255, 0, 0)).save(path)
    return path


@pytest.fixture
def mock_editor(monkeypatch):
    """Replace ImageEditor with a fake that writes a 1x1 PNG."""
    editor_instances = []

    def fake_ctor(*, model_name, quantize=None):
        inst = MagicMock()
        inst.model_name = model_name
        inst.quantize = quantize

        def fake_edit(*, image_paths, prompt, output_path, **kwargs):
            Image.new("RGB", (1, 1), (255, 255, 255)).save(output_path)

        inst.edit = MagicMock(side_effect=fake_edit)
        editor_instances.append(inst)
        return inst

    monkeypatch.setattr("fluxgen_mcp.tools.edit.ImageEditor", fake_ctor)
    return editor_instances


pytestmark = pytest.mark.asyncio


async def test_edit_basic(tmp_path: Path, mock_editor):
    s = _settings(tmp_path)
    inp = _make_png(tmp_path / "input.png", (512, 512))
    result = await edit_image_tool(
        settings=s,
        input_paths=[str(inp)],
        prompt="make it sunset",
        model=None,
        seed=42,
        steps=None,
        guidance=None,
        width=None,
        height=None,
        output_subdir="edits",
    )
    assert result["model"] == "flux2-klein"
    assert Path(result["path"]).exists()
    assert len(mock_editor) == 1
    assert mock_editor[0].edit.call_args.kwargs["prompt"] == "make it sunset"
    assert mock_editor[0].edit.call_args.kwargs["seed"] == 42


async def test_edit_rejects_model_not_in_whitelist(tmp_path: Path, mock_editor):
    s = _settings(tmp_path, allowed_edit_models=("flux2-klein",))
    inp = _make_png(tmp_path / "input.png", (100, 100))
    with pytest.raises(MCPError) as exc:
        await edit_image_tool(
            settings=s,
            input_paths=[str(inp)],
            prompt="x",
            model="qwen-image-edit",
            seed=None,
            steps=None,
            guidance=None,
            width=None,
            height=None,
            output_subdir="default",
        )
    assert exc.value.code == E_BAD_ARG


async def test_edit_rejects_multiple_inputs_for_qwen(tmp_path: Path, mock_editor):
    s = _settings(tmp_path)
    inp1 = _make_png(tmp_path / "a.png", (100, 100))
    inp2 = _make_png(tmp_path / "b.png", (100, 100))
    with pytest.raises(MCPError) as exc:
        await edit_image_tool(
            settings=s,
            input_paths=[str(inp1), str(inp2)],
            prompt="x",
            model="qwen-image-edit",
            seed=None,
            steps=None,
            guidance=None,
            width=None,
            height=None,
            output_subdir="default",
        )
    assert exc.value.code == E_BAD_ARG


async def test_edit_rejects_oversized_input(tmp_path: Path, mock_editor):
    s = _settings(tmp_path, input_max_bytes=10)
    inp = _make_png(tmp_path / "input.png", (100, 100))
    inp.write_bytes(b"x" * 100)  # exceeds the 10-byte cap
    with pytest.raises(MCPError) as exc:
        await edit_image_tool(
            settings=s,
            input_paths=[str(inp)],
            prompt="x",
            model=None,
            seed=None,
            steps=None,
            guidance=None,
            width=None,
            height=None,
            output_subdir="default",
        )
    assert exc.value.code == E_INPUT_TOO_LARGE


async def test_edit_rejects_missing_input(tmp_path: Path, mock_editor):
    """Missing input file should raise E_INVALID_INPUT_IMAGE, not
    bubble as E_INTERNAL via the server's bare-Exception fallback.
    """
    s = _settings(tmp_path)
    with pytest.raises(MCPError) as exc:
        await edit_image_tool(
            settings=s,
            input_paths=[str(tmp_path / "missing.png")],
            prompt="x",
            model=None,
            seed=None,
            steps=None,
            guidance=None,
            width=None,
            height=None,
            output_subdir="default",
        )
    assert exc.value.code == E_INVALID_INPUT_IMAGE


async def test_edit_rejects_steps_over_cap(tmp_path: Path, mock_editor):
    s = _settings(tmp_path, max_steps=10)
    inp = _make_png(tmp_path / "input.png", (100, 100))
    with pytest.raises(MCPError) as exc:
        await edit_image_tool(
            settings=s,
            input_paths=[str(inp)],
            prompt="x",
            model=None,
            seed=None,
            steps=99,
            guidance=None,
            width=None,
            height=None,
            output_subdir="default",
        )
    assert exc.value.code == E_BAD_ARG


async def test_edit_empty_input_paths(tmp_path: Path, mock_editor):
    s = _settings(tmp_path)
    with pytest.raises(MCPError) as exc:
        await edit_image_tool(
            settings=s,
            input_paths=[],
            prompt="x",
            model=None,
            seed=None,
            steps=None,
            guidance=None,
            width=None,
            height=None,
            output_subdir="default",
        )
    assert exc.value.code == E_BAD_ARG


async def test_edit_maps_known_exception_to_e_model(tmp_path: Path, monkeypatch, mock_editor):
    """A `FluxgenError` raised by `editor.edit()` flows through the
    `EXCEPTION_MAP` and surfaces as `MCPError(E_MODEL)` with seed +
    model attached.
    """
    def boom(*, image_paths, prompt, output_path, **kwargs):
        raise FluxgenError("edit pipeline failed")

    monkeypatch.setattr(
        "fluxgen_mcp.tools.edit.ImageEditor",
        lambda *, model_name, quantize=None: MagicMock(
            model_name=model_name,
            quantize=quantize,
            _load_pipeline=lambda: None,
            edit=MagicMock(side_effect=boom),
        ),
    )
    s = _settings(tmp_path)
    inp = _make_png(tmp_path / "input.png", (100, 100))
    with pytest.raises(MCPError) as exc:
        await edit_image_tool(
            settings=s,
            input_paths=[str(inp)],
            prompt="x",
            model=None,
            seed=11,
            steps=None,
            guidance=None,
            width=None,
            height=None,
            output_subdir="default",
        )
    assert exc.value.code == E_MODEL
    assert exc.value.seed == 11


async def test_edit_maps_oserror_to_e_model(tmp_path: Path, monkeypatch, mock_editor):
    """`OSError` raised inside `_load_pipeline()` (e.g. model file
    missing) is caught by the widening clause and surfaces as
    `E_MODEL` rather than `E_INTERNAL`.
    """
    def load_boom(self=None):
        raise OSError("model file not found")

    monkeypatch.setattr(
        "fluxgen_mcp.tools.edit.ImageEditor",
        lambda *, model_name, quantize=None: MagicMock(
            model_name=model_name,
            quantize=quantize,
            _load_pipeline=MagicMock(side_effect=load_boom),
            edit=MagicMock(),
        ),
    )
    s = _settings(tmp_path)
    inp = _make_png(tmp_path / "input.png", (100, 100))
    with pytest.raises(MCPError) as exc:
        await edit_image_tool(
            settings=s,
            input_paths=[str(inp)],
            prompt="x",
            model=None,
            seed=None,
            steps=None,
            guidance=None,
            width=None,
            height=None,
            output_subdir="default",
        )
    assert exc.value.code == E_MODEL