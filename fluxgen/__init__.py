"""fluxgen-cli: local AI image generation and editing on Apple Silicon.

Defines the public package version via ``importlib.metadata`` so the
CLI's ``--version`` output and any embedder import the same source of
truth. Falls back to a literal on source-only checkouts where the
distribution metadata is unavailable (e.g., before ``uv sync``).

Keep the literal in sync with ``[project] version`` in pyproject.toml.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("fluxgen-cli")
except PackageNotFoundError:
    __version__ = "0.4.0"

__all__ = ["__version__"]
