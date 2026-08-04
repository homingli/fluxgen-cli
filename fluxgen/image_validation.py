"""Validate input image files for the `generate` and `edit` flows.

Centralizes the existence + is_file + PIL integrity checks so the
generation flow's img2img path and the editing flow's input-image
path cannot drift apart.

This helper is intentionally a *validate-image-input* helper, not a
generic image I/O utility. Callers in the `gen` and `edit` paths
delegate user-facing input validation here. Future flows that need
different semantics (upscaling, embedding, OCR, etc.) should add
their own helpers rather than reusing this one.
"""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from fluxgen.exceptions import InvalidImageError


def validate_image_file(
    path: str | os.PathLike[str],
    *,
    read_size: bool = False,
    label: str = "image",
) -> Path | tuple[Path, tuple[int, int]]:
    """Resolve, existence-check, and PIL-integrity-verify an image file.

    This is the input-side validation helper for the `gen` (img2img
    `--init-image`) and `edit` (positional `image` args) commands.

    Args:
        path: Filesystem path to the image. ``~`` and relative
            components are expanded and resolved before any filesystem
            call.
        read_size: When True, additionally return the image's
            ``(width, height)`` derived from the PIL header. The size
            lookup and ``Image.verify()`` integrity check both happen
            inside the same ``Image.open()`` block, so the file is
            opened exactly once regardless of ``read_size``.
        label: Short noun used in the ``FileNotFoundError`` /
            ``ValueError`` messages. Defaults to ``"image"``; the
            edit path passes ``"input image"`` and the gen img2img
            path passes ``"reference image"`` to keep the original
            pre-refactor messages. Caller must pass lowercase; the
            helper applies ``str.capitalize``.

    Returns:
        When ``read_size=False``: the resolved ``pathlib.Path``.
        When ``read_size=True``: a 2-tuple
        ``(resolved_path: Path, size: tuple[int, int])``.

    Raises:
        FileNotFoundError: path does not exist. Message includes the
            ``label`` and the resolved path.
        ValueError: path exists but is not a regular file (e.g., a
            directory or a symlink to one).
        InvalidImageError: file is unreadable or fails the PIL
            integrity check.

    Note:
        ``Image.verify()`` closes the underlying file handle and
        invalidates any cached pixel data, so the helper returns
        metadata only. **Callers that need pixel data (e.g., the
        edit path's RGB conversion) must reopen the file.** See
        ``editor.py`` for the reopen pattern.
    """
    resolved = Path(path).expanduser().resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"{label.capitalize()} not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{label.capitalize()} must be a file: {resolved}")

    # Open once. `.size` is a free header lookup; it MUST be read
    # before `verify()` because `verify()` walks the file and
    # invalidates the cached pixel data, so any later `img.size`
    # call would have to reopen. After this block returns, the file
    # handle is closed and the pixel cache is gone; callers that
    # need pixel data must reopen. See `editor.py` for the reopen.
    size = None
    try:
        with Image.open(resolved) as img:
            if read_size:
                size = img.size
            img.verify()
    except UnidentifiedImageError:
        raise InvalidImageError(
            f"Invalid or corrupted image file: {resolved}"
        )
    except (OSError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(
            f"Could not verify image file {resolved}: {exc}"
        )

    if read_size:
        return resolved, size
    return resolved
