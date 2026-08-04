"""Image input validation for the MCP tool layer.

Wraps `fluxgen.image_validation.validate_image_file` with the
MCP-specific bounds (size cap, dimension cap) and emits `MCPError`
on rejection. The base validator raises `fluxgen.exceptions` which
the tool layer maps via `EXCEPTION_MAP`; the size and dimension
checks raise `MCPError` directly because they are MCP-policy
overrides on top of the underlying image validity.
"""
from __future__ import annotations

import os
from pathlib import Path

from fluxgen.image_validation import validate_image_file
from fluxgen_mcp.errors import (
    E_BAD_ARG,
    E_INVALID_INPUT_IMAGE,
    E_INPUT_TOO_HIGHRES,
    E_INPUT_TOO_LARGE,
    MCPError,
)


def validate_edit_inputs(
    paths: list[str | os.PathLike[str]],
    *,
    max_bytes: int,
    max_dimension: int,
) -> list[Path]:
    """Validate image inputs for `edit_image` and `generate_image.init_image_path`.

    For each path:
      1. Short-circuit on missing path → `E_INVALID_INPUT_IMAGE`
         (avoids the `FileNotFoundError → E_INTERNAL` reroute through
         `validate_image_file`).
      2. Size cap (cheap, before opening the file).
      3. PIL integrity + read size in one pass
         (`validate_image_file(read_size=True)` returns
         `(Path, (w, h))`).
      4. Width / height cap.

    Returns the resolved `Path` list. Order is preserved.

    Args:
        paths: Image paths (str or os.PathLike).
        max_bytes: Reject files larger than this (bytes).
        max_dimension: Reject images whose longest side exceeds this.

    Returns:
        List of resolved `Path` objects.

    Raises:
        MCPError(E_INVALID_INPUT_IMAGE): path does not exist.
        MCPError(E_INPUT_TOO_LARGE): any path's file size exceeds
            `max_bytes`.
        MCPError(E_INPUT_TOO_HIGHRES): any image's width or height
            exceeds `max_dimension`.
        FileNotFoundError / ValueError / InvalidImageError: from the
            underlying validator; the tool layer maps these to
            `E_INVALID_INPUT_IMAGE`.
    """
    if not paths:
        raise MCPError(E_BAD_ARG, "no input images provided")

    resolved: list[Path] = []
    for p in paths:
        path = Path(p).expanduser()

        # Short-circuit missing files. Without this, the underlying
        # validator raises FileNotFoundError, which is mapped to
        # E_INVALID_INPUT_IMAGE via EXCEPTION_MAP, but the message
        # loses the path. Catching here keeps the path in the error.
        if not path.exists():
            raise MCPError(
                E_INVALID_INPUT_IMAGE,
                f"input path does not exist: {path}",
            )

        # Size check before opening — keeps us from allocating MB of
        # memory for an obviously-oversized file. Race window
        # between stat() and the underlying open is small and the
        # validator's FileNotFoundError still maps correctly.
        size = path.stat().st_size
        if size > max_bytes:
            raise MCPError(
                E_INPUT_TOO_LARGE,
                f"input {path} is {size} bytes; max {max_bytes}",
            )

        validated, (w, h) = validate_image_file(path, read_size=True, label="input image")

        if max(w, h) > max_dimension:
            raise MCPError(
                E_INPUT_TOO_HIGHRES,
                f"input {validated} is {w}x{h}; max dimension {max_dimension}",
            )

        resolved.append(validated)

    return resolved


def validate_init_image(
    path: str | os.PathLike[str],
    *,
    max_bytes: int,
    max_dimension: int,
) -> Path:
    """Validate the single `init_image_path` argument for `generate_image`.

    Thin wrapper that delegates to `validate_edit_inputs` with a
    one-element list. Kept separate so the schema of `generate_image`
    reads cleanly.
    """
    result = validate_edit_inputs([path], max_bytes=max_bytes, max_dimension=max_dimension)
    return result[0]