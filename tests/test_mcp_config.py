"""Tests for fluxgen_mcp.config."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from fluxgen_mcp.config import DEFAULTS, MCPSettings, load_mcp_settings


def test_defaults_when_no_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    s = load_mcp_settings()
    assert s.max_width == DEFAULTS["max_width"]
    assert s.max_height == DEFAULTS["max_height"]
    assert s.allowed_generation_models == DEFAULTS["allowed_generation_models"]
    assert s.input_max_bytes == 20 * 1024 * 1024
    assert s.input_max_dimension == 1080


def test_cwd_overrides_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    home.mkdir()
    cwd.mkdir()
    (home / ".fluxgen.toml").write_text(
        textwrap.dedent(
            """\
            [mcp]
            max_width = 800
            max_height = 800
            """
        )
    )
    (cwd / ".fluxgen.toml").write_text(
        textwrap.dedent(
            """\
            [mcp]
            max_width = 1024
            """
        )
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(cwd)
    s = load_mcp_settings()
    assert s.max_width == 1024  # cwd wins
    assert s.max_height == 800  # home's value still applies


def test_invalid_toml_does_not_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".fluxgen.toml").write_text("not valid toml [[[")
    s = load_mcp_settings()  # should log and fall back to defaults
    assert s.max_width == DEFAULTS["max_width"]


def test_expand_paths_expands_tilde(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    s = MCPSettings(
        output_root="~/foo",
        max_width=1,
        max_height=1,
        max_steps=1,
        max_prompt_chars=1,
        max_concurrent_jobs=1,
        max_queue_depth=0,
        per_call_timeout_s=1.0,
        allowed_generation_models=(),
        allowed_edit_models=(),
        prompt_blocklist=(),
        audit_log_path="~/audit.log",
        pause_sentinel_path="~/paused",
        pid_file_path="~/pid",
        input_max_bytes=1,
        input_max_dimension=1,
    )
    expanded = s.expand_paths()
    assert expanded.output_root == str(tmp_path / "foo")
    assert expanded.audit_log_path == str(tmp_path / "audit.log")
    # Non-path fields unchanged
    assert expanded.max_width == 1


def test_load_mcp_settings_precompiles_blocklist(tmp_path, monkeypatch):
    """Patterns in `.fluxgen.toml` are strings; `load_mcp_settings`
    must precompile them to `re.Pattern` so `validate_prompt` does
    not pay `re.compile` per call.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".fluxgen.toml").write_text(
        textwrap.dedent(
            """\
            [mcp]
            prompt_blocklist = ["foo", "bar"]
            """
        )
    )
    s = load_mcp_settings()
    import re

    assert len(s.prompt_blocklist) == 2
    assert all(isinstance(p, re.Pattern) for p in s.prompt_blocklist)
    # Pre-compiled patterns are case-insensitive (per config contract).
    assert any(p.search("FOO") for p in s.prompt_blocklist)


def test_load_mcp_settings_drops_invalid_regex(tmp_path, monkeypatch):
    """A bad regex in `[mcp].prompt_blocklist` must NOT crash config
    load. `load_mcp_settings` drops it with a warning; the server
    runs with the surviving patterns.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".fluxgen.toml").write_text(
        textwrap.dedent(
            """\
            [mcp]
            prompt_blocklist = ["good", "[invalid"]
            """
        )
    )
    s = load_mcp_settings()
    # Only the valid pattern survives.
    assert len(s.prompt_blocklist) == 1
    assert s.prompt_blocklist[0].search("good")
    assert not s.prompt_blocklist[0].search("[invalid")
