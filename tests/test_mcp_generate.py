"""Tests for the generate_image tool wrapper.

The underlying `fluxgen.generator.generate_image` is mocked at the
import seam so the tests don't load mflux / torch.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from fluxgen_mcp.config import MCPSettings
from fluxgen_mcp.errors import (
    E_BAD_ARG,
    E_INVALID_INPUT_IMAGE,
    E_PATH_TRAVERSAL,
    E_PROMPT_TOO_LONG,
    MCPError,
)
from fluxgen_mcp.tools.generate import generate_image_tool


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
        allowed_generation_models=("zimage-turbo", "zimage"),
        allowed_edit_models=("flux2-klein",),
        prompt_blocklist=(),
        audit_log_path=str(tmp_path / "audit.log"),
        pause_sentinel_path=str(tmp_path / "paused"),
        pid_file_path=str(tmp_path / "pid"),
        input_max_bytes=20_000_000,
        input_max_dimension=1080,
    )
    base.update(overrides)
    return MCPSettings(**base)


@pytest.fixture
def mock_generate(monkeypatch):
    """Replace fluxgen.generator.generate_image with a fake.

    Records calls and writes a 1x1 PNG so downstream validation
    (file existence, format) doesn't fail.
    """
    calls = []

    def fake_generate(*, prompt, preset, seed, output, width, height,
                      style, init_image, strength, model_name, **_):
        calls.append({
            "prompt": prompt,
            "preset": preset,
            "seed": seed,
            "output": output,
            "width": width,
            "height": height,
            "style": style,
            "init_image": init_image,
            "strength": strength,
            "model_name": model_name,
        })
        Image.new("RGB", (1, 1), (255, 255, 255)).save(output)

    monkeypatch.setattr("fluxgen_mcp.tools.generate.generate_image", fake_generate)
    return calls


pytestmark = pytest.mark.asyncio


async def test_generate_basic(tmp_path: Path, mock_generate):
    s = _settings(tmp_path)
    result = await generate_image_tool(
        settings=s,
        prompt="a cat",
        model=None,
        preset=None,
        width=None,
        height=None,
        seed=None,
        style=None,
        init_image_path=None,
        strength=None,
        output_subdir="cats",
    )
    assert result["model"] == "zimage-turbo"
    assert result["width"] == 512
    assert result["height"] == 512
    assert "cats" in result["path"]
    assert Path(result["path"]).exists()
    assert len(mock_generate) == 1
    assert mock_generate[0]["prompt"] == "a cat"
    assert mock_generate[0]["strength"] == 0.4  # default


async def test_generate_with_seed_and_dimensions(tmp_path: Path, mock_generate):
    s = _settings(tmp_path)
    result = await generate_image_tool(
        settings=s,
        prompt="a fox",
        model="zimage",
        preset="quality",
        width=1024,
        height=1024,
        seed=42,
        style="cinematic",
        init_image_path=None,
        strength=0.7,
        output_subdir="default",
    )
    assert result["seed"] == 42
    assert mock_generate[0]["seed"] == 42
    assert mock_generate[0]["model_name"] == "zimage"
    assert mock_generate[0]["width"] == 1024
    assert mock_generate[0]["height"] == 1024
    assert mock_generate[0]["strength"] == 0.7


async def test_generate_rejects_model_not_in_whitelist(tmp_path: Path, mock_generate):
    s = _settings(tmp_path)
    with pytest.raises(MCPError) as exc:
        await generate_image_tool(
            settings=s,
            prompt="a cat",
            model="nonexistent-model",
            preset=None,
            width=None,
            height=None,
            seed=None,
            style=None,
            init_image_path=None,
            strength=None,
            output_subdir="default",
        )
    assert exc.value.code == E_BAD_ARG


async def test_generate_rejects_dimensions_over_cap(tmp_path: Path, mock_generate):
    s = _settings(tmp_path, max_width=800)
    with pytest.raises(MCPError) as exc:
        await generate_image_tool(
            settings=s,
            prompt="a cat",
            model=None,
            preset=None,
            width=1024,
            height=None,
            seed=None,
            style=None,
            init_image_path=None,
            strength=None,
            output_subdir="default",
        )
    assert exc.value.code == E_BAD_ARG


async def test_generate_rejects_prompt_too_long(tmp_path: Path, mock_generate):
    s = _settings(tmp_path, max_prompt_chars=10)
    with pytest.raises(MCPError) as exc:
        await generate_image_tool(
            settings=s,
            prompt="x" * 100,
            model=None,
            preset=None,
            width=None,
            height=None,
            seed=None,
            style=None,
            init_image_path=None,
            strength=None,
            output_subdir="default",
        )
    assert exc.value.code == E_PROMPT_TOO_LONG


async def test_generate_rejects_path_traversal(tmp_path: Path, mock_generate):
    s = _settings(tmp_path)
    with pytest.raises(MCPError) as exc:
        await generate_image_tool(
            settings=s,
            prompt="a cat",
            model=None,
            preset=None,
            width=None,
            height=None,
            seed=None,
            style=None,
            init_image_path=None,
            strength=None,
            output_subdir="../../etc",
        )
    assert exc.value.code == E_PATH_TRAVERSAL


async def test_generate_respects_pause(tmp_path: Path, mock_generate):
    s = _settings(tmp_path)
    sentinel = tmp_path / "paused"
    sentinel.touch()
    with pytest.raises(MCPError) as exc:
        await generate_image_tool(
            settings=s,
            prompt="a cat",
            model=None,
            preset=None,
            width=None,
            height=None,
            seed=None,
            style=None,
            init_image_path=None,
            strength=None,
            output_subdir="default",
        )
    assert exc.value.code == "E_DISABLED"
    assert len(mock_generate) == 0


async def test_generate_attach_meta_to_mcperror(tmp_path: Path, monkeypatch, mock_generate):
    """Tool-layer errors must carry seed + model + output_path so the
    server's audit writer can populate them."""
    from fluxgen.exceptions import FluxgenError

    def boom(*, prompt, preset, seed, output, width, height, style,
             init_image, strength, model_name, **kwargs):
        raise FluxgenError("model load failed")

    monkeypatch.setattr("fluxgen_mcp.tools.generate.generate_image", boom)
    s = _settings(tmp_path)
    with pytest.raises(MCPError) as exc:
        await generate_image_tool(
            settings=s,
            prompt="a cat",
            model=None,
            preset=None,
            width=None,
            height=None,
            seed=99,
            style=None,
            init_image_path=None,
            strength=None,
            output_subdir="default",
        )
    assert exc.value.seed == 99
    assert exc.value.model == "zimage-turbo"
    assert exc.value.output_path is None


async def test_generate_maps_filenotfounderror_init(tmp_path: Path, mock_generate):
    """If the init_image_path does not exist, the tool layer raises
    `E_INVALID_INPUT_IMAGE` (not `E_INTERNAL`) — the validation.py
    short-circuit fires before `validate_image_file`.
    """
    s = _settings(tmp_path)
    with pytest.raises(MCPError) as exc:
        await generate_image_tool(
            settings=s,
            prompt="a cat",
            model=None,
            preset=None,
            width=None,
            height=None,
            seed=None,
            style=None,
            init_image_path=str(tmp_path / "missing.png"),
            strength=None,
            output_subdir="default",
        )
    assert exc.value.code == E_INVALID_INPUT_IMAGE