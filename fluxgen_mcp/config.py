"""Configuration loading for the MCP server.

Reads the `[mcp]` section of `.fluxgen.toml` from the current working
directory first, then the user's home directory (later wins, matching
`fluxgen.config.load_config` semantics).

This module only owns MCP-specific keys. The CLI section is left to
`fluxgen.config`.
"""
from __future__ import annotations

import logging
import re
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Pattern

logger = logging.getLogger("fluxgen-mcp")

DEFAULT_CONFIG_FILENAME = ".fluxgen.toml"

# Hard-coded defaults. Override per-deployment by adding `[mcp]` to a
# `.fluxgen.toml` (cwd takes precedence over home).
DEFAULTS = {
    "output_root": "~/fluxgen-mcp-output",
    "max_width": 1920,
    "max_height": 1920,
    "max_steps": 50,
    "max_prompt_chars": 2000,
    "max_concurrent_jobs": 1,
    "max_queue_depth": 4,
    "per_call_timeout_s": 600.0,
    "allowed_generation_models": (
        "zimage-turbo",
        "zimage",
        "flux2-klein4b",
        "flux2-klein9b",
    ),
    "allowed_edit_models": ("flux2-klein", "qwen-image-edit"),
    "prompt_blocklist": (),
    "audit_log_path": "~/.fluxgen-mcp-audit.log",
    "pause_sentinel_path": "~/.fluxgen-mcp-paused",
    "pid_file_path": "~/.fluxgen-mcp.pid",
    "input_max_bytes": 20 * 1024 * 1024,
    "input_max_dimension": 1080,
}


@dataclass(frozen=True)
class MCPSettings:
    """Resolved MCP server settings.

    `path` fields are kept as raw strings (`~/...`) until they are
    expanded lazily by `expand_paths`. Callers should use the expanded
    fields when reading from disk; the raw fields are useful for
    snapshotting config in error messages.
    """

    output_root: str
    max_width: int
    max_height: int
    max_steps: int
    max_prompt_chars: int
    max_concurrent_jobs: int
    max_queue_depth: int
    per_call_timeout_s: float
    allowed_generation_models: tuple[str, ...]
    allowed_edit_models: tuple[str, ...]
    audit_log_path: str
    pause_sentinel_path: str
    pid_file_path: str
    input_max_bytes: int
    input_max_dimension: int
    # Patterns are precompiled at config-load time (see
    # `load_mcp_settings`); the field defaults to an empty tuple so
    # callers constructing `MCPSettings` directly don't need to
    # compile.
    prompt_blocklist: tuple[Pattern[str], ...] = field(default=())

    def expand_paths(self) -> "MCPSettings":
        """Return a copy with all path fields expanded and resolved.

        Does NOT touch non-path fields. The expanded `Path` objects
        are converted back to strings via `str(...)` so the rest of
        the code can keep treating `MCPSettings` as a plain config
        object.
        """

        def _expand(p: str) -> str:
            return str(Path(p).expanduser().resolve())

        return replace(
            self,
            output_root=_expand(self.output_root),
            audit_log_path=_expand(self.audit_log_path),
            pause_sentinel_path=_expand(self.pause_sentinel_path),
            pid_file_path=_expand(self.pid_file_path),
        )


def _coerce(value, default):
    """Coerce a TOML-decoded value to the default's type.

    Lists/tuples go through `tuple(...)`. Tuples in TOML become lists;
    convert them back so downstream code can rely on immutability.
    Booleans, ints, floats, strings are returned as-is.
    """
    if isinstance(default, tuple):
        return tuple(value) if value is not None else default
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, str):
        return str(value)
    return value


def load_mcp_settings() -> MCPSettings:
    """Load `[mcp]` from `.fluxgen.toml` in cwd and home.

    Returns a frozen `MCPSettings` with all keys present. Missing
    keys fall back to `DEFAULTS`. The `prompt_blocklist` patterns
    are precompiled (case-insensitive) at load time so per-call
    validation does not pay `re.compile` cost.
    """
    merged: dict = dict(DEFAULTS)

    locations = [
        Path.home() / DEFAULT_CONFIG_FILENAME,
        Path.cwd() / DEFAULT_CONFIG_FILENAME,
    ]

    for loc in locations:
        if not loc.exists():
            continue
        try:
            with open(loc, "rb") as f:
                data = tomllib.load(f)
        except (tomllib.TOMLDecodeError, OSError) as exc:
            logger.warning("failed to load MCP config from %s: %s", loc, exc)
            continue
        section = data.get("mcp")
        if not isinstance(section, dict):
            continue
        for key, default in DEFAULTS.items():
            if key in section:
                merged[key] = _coerce(section[key], default)

    # Precompile blocklist patterns. Invalid regex strings are
    # dropped here (with a warning) so a typo in the config does
    # not crash the server.
    raw_patterns: tuple[str, ...] = merged["prompt_blocklist"]
    compiled: list[Pattern[str]] = []
    for pattern in raw_patterns:
        try:
            compiled.append(re.compile(pattern, flags=re.IGNORECASE))
        except re.error as exc:
            logger.warning(
                "invalid prompt_blocklist regex %r: %s", pattern, exc,
            )
    merged["prompt_blocklist"] = tuple(compiled)

    return MCPSettings(**merged)
