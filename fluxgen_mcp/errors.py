"""Error type for the MCP layer.

`MCPError` is the single in-process exception every tool raises. The
code is preserved on the instance and emitted as the error message
prefix so MCP clients can branch on it. The MCP server wraps anything
else as `ToolError`; we deliberately do not subclass `ToolError`
itself so we don't need to import from `mcp.server.mcpserver.exceptions`
in the inner layers (kept as a thin façade to ease future API
upgrades).
"""
from __future__ import annotations

from typing import Any


class MCPError(Exception):
    """An error raised by a fluxgen-mcp tool.

    Attributes:
        code: Machine-readable code (e.g. ``"E_PATH_TRAVERSAL"``).
        message: Human-readable detail, safe to surface to the agent.
        seed / model / output_path: Optional context forwarded by the
            tool layer so the server can build a complete audit
            record. Stored as instance attributes, not in `args`,
            so they never reach the MCP error message.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        seed: int | None = None,
        model: str | None = None,
        output_path: str | None = None,
    ):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.seed = seed
        self.model = model
        self.output_path = output_path

    def audit_meta(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "model": self.model,
            "output_path": self.output_path,
        }


# Canonical error codes. Single source of truth — referenced from
# docs and asserted in tests.
E_DISABLED = "E_DISABLED"
E_PATH_TRAVERSAL = "E_PATH_TRAVERSAL"
E_PROMPT_TOO_LONG = "E_PROMPT_TOO_LONG"
E_PROMPT_REJECTED = "E_PROMPT_REJECTED"
E_INVALID_INPUT_IMAGE = "E_INVALID_INPUT_IMAGE"
E_INPUT_TOO_LARGE = "E_INPUT_TOO_LARGE"
E_INPUT_TOO_HIGHRES = "E_INPUT_TOO_HIGHRES"
E_BAD_ARG = "E_BAD_ARG"
E_MODEL = "E_MODEL"
E_TIMEOUT = "E_TIMEOUT"
E_BUSY = "E_BUSY"
E_INTERNAL = "E_INTERNAL"


# Map `fluxgen` and stdlib exceptions to MCP error codes. The MCP
# wrapper never raises these directly — it catches them in the tool
# layer and re-raises as `MCPError`.
from fluxgen.exceptions import (  # noqa: E402
    FluxgenError,
    InvalidImageError,
    InvalidConfigurationError,
    PathTraversalError,
    ModelLoadError,
)


EXCEPTION_MAP: dict[type[Exception], str] = {
    PathTraversalError: E_PATH_TRAVERSAL,
    InvalidImageError: E_INVALID_INPUT_IMAGE,
    # validate_image_file raises these for missing files / corrupt
    # inputs. Without this entry the server's `except Exception`
    # catches them and surfaces as E_INTERNAL, masking the real
    # cause from the agent.
    FileNotFoundError: E_INVALID_INPUT_IMAGE,
    ValueError: E_INVALID_INPUT_IMAGE,
    InvalidConfigurationError: E_BAD_ARG,
    ModelLoadError: E_MODEL,
    FluxgenError: E_MODEL,
}