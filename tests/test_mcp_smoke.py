"""Smoke tests for `fluxgen_mcp` import surface.

These guard against future `mcp` releases renaming or moving the
seams we depend on (`MCPServer`, `Context`, `ToolError`). If a
release breaks any of these imports, the smoke test fails before
the actual call paths get exercised, which is the value of a
"import everything" test rather than a full functional one.
"""
from __future__ import annotations

import importlib
import sys

import pytest


def test_fluxgen_mcp_package_imports():
    mod = importlib.import_module("fluxgen_mcp")
    assert mod.__version__


def test_server_module_imports():
    """The full server import chain must succeed — the `mcp`
    package is a soft dependency, so an ImportError here means
    the install was incomplete.
    """
    if "fluxgen_mcp.server" in sys.modules:
        importlib.reload(sys.modules["fluxgen_mcp.server"])
    else:
        importlib.import_module("fluxgen_mcp.server")


def test_mcp_seams_are_present():
    """Verify the exact symbols we depend on exist on the installed
    `mcp` package. Pin `mcp>=1.2,<2.0` in pyproject.toml so this
    contract is enforced by the resolver.
    """
    from mcp.server.mcpserver import Context, MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    assert MCPServer is not None
    assert Context is not None
    assert ToolError is not None


def test_mcp_version_is_pinned_range():
    """`mcp>=2.0,<3.0` per pyproject.toml. The 2.0 release renamed
    `FastMCP` to `MCPServer` and moved it under
    `mcp.server.mcpserver`; this package depends on the 2.x API.
    We check at runtime as a secondary guard against an accidental
    resolver change.
    """
    from importlib.metadata import version

    v = version("mcp")
    parts = v.split(".")
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 else 0
    assert (major, minor) >= (2, 0), (
        f"mcp {v} is below the minimum 2.0; 1.x FastMCP API is not supported"
    )
    assert major < 3, f"mcp {v} is outside the pinned 2.x range"