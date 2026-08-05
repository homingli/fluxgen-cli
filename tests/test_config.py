"""Tests for fluxgen.config: loading + schema validation."""
import logging
from pathlib import Path

import pytest

from fluxgen.config import (
    DEFAULT_CONFIG_FILENAME,
    _KNOWN_DEFAULT_KEYS,
    get_config_value,
    load_config,
)


# ── Schema allow-list contract ──────────────────────────────────────────────


def test_known_default_keys_is_frozenset():
    """The schema allow-list is immutable so accidental mutation can't widen it."""
    assert isinstance(_KNOWN_DEFAULT_KEYS, frozenset)


def test_known_default_keys_contains_expected():
    """Documented schema keys are present so the contract is reviewable."""
    expected_subset = {
        "model",
        "output_dir",
        "width",
        "height",
        "preset",
        "style",
        "max_edit_dimension",
    }
    assert expected_subset <= _KNOWN_DEFAULT_KEYS


def test_known_default_keys_excludes_unrelated():
    """StyleManager settings are NOT in [defaults]; they belong in [styles]."""
    assert "retro" not in _KNOWN_DEFAULT_KEYS
    assert "ghibli" not in _KNOWN_DEFAULT_KEYS


# ── load_config: empty / missing ─────────────────────────────────────────────


def test_load_config_returns_empty_when_no_files(monkeypatch, tmp_path):
    """No .fluxgen.toml → empty dict, no errors."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert load_config() == {}


# ── load_config: unknown key rejection ───────────────────────────────────────


def test_load_config_drops_unknown_default_keys(monkeypatch, tmp_path, caplog):
    """Unknown [defaults] keys are dropped; warning is emitted."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / DEFAULT_CONFIG_FILENAME
    cfg.write_text(
        '[defaults]\n'
        'model = "zimage-turbo"\n'
        'unknown_typo_key = "should be dropped"\n'
    )

    with caplog.at_level(logging.WARNING, logger="fluxgen"):
        config = load_config()

    assert config["defaults"]["model"] == "zimage-turbo"
    assert "unknown_typo_key" not in config["defaults"]
    assert any(
        "unknown_typo_key" in rec.message and "[defaults]" in rec.message
        for rec in caplog.records
    )


def test_load_config_keeps_all_known_default_keys(monkeypatch, tmp_path):
    """Every key in the schema allow-list survives a round-trip."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / DEFAULT_CONFIG_FILENAME
    cfg.write_text(
        '[defaults]\n'
        'model = "zimage"\n'
        'output_dir = "out"\n'
        'width = 1024\n'
        'height = 768\n'
        'preset = 3\n'
        'style = "ghibli"\n'
        'max_edit_dimension = 1024\n'
    )

    config = load_config()
    assert config["defaults"] == {
        "model": "zimage",
        "output_dir": "out",
        "width": 1024,
        "height": 768,
        "preset": 3,
        "style": "ghibli",
        "max_edit_dimension": 1024,
    }


def test_load_config_styles_section_is_open_ended(monkeypatch, tmp_path, caplog):
    """[styles] is user-defined; unknown keys there are NOT warned."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / DEFAULT_CONFIG_FILENAME
    cfg.write_text('[styles]\nretro = " in 80s style"\ncustom = " custom suffix"\n')

    with caplog.at_level(logging.WARNING, logger="fluxgen"):
        config = load_config()

    assert config["styles"] == {"retro": " in 80s style", "custom": " custom suffix"}
    # No 'unknown key' warning should be emitted for the styles block.
    assert not any(
        "unknown key" in rec.message.lower() for rec in caplog.records
    )


# ── load_config: merging & error handling ────────────────────────────────────


def test_load_config_cwd_overrides_home(monkeypatch, tmp_path):
    """cwd .fluxgen.toml wins over HOME .fluxgen.toml for known keys."""
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    home.mkdir()
    cwd.mkdir()
    home_cfg = home / DEFAULT_CONFIG_FILENAME
    home_cfg.write_text('[defaults]\nmodel = "zimage-turbo"\nstyle = "ghibli"\n')
    cwd_cfg = cwd / DEFAULT_CONFIG_FILENAME
    cwd_cfg.write_text('[defaults]\nmodel = "zimage"\n')

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(cwd)
    config = load_config()

    assert config["defaults"]["model"] == "zimage"  # cwd wins
    assert config["defaults"]["style"] == "ghibli"  # inherited from home


def test_load_config_invalid_toml_warns_and_continues(monkeypatch, tmp_path, caplog):
    """Malformed TOML in one file does not abort the whole load."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    bad_cfg = tmp_path / DEFAULT_CONFIG_FILENAME
    bad_cfg.write_text("[defaults\nthis is not valid toml")

    with caplog.at_level(logging.WARNING, logger="fluxgen"):
        config = load_config()

    assert config == {}
    assert any("Failed to load config" in rec.message for rec in caplog.records)


# ── get_config_value ────────────────────────────────────────────────────────


def test_get_config_value_returns_default_when_missing():
    assert get_config_value({}, "model", "zimage-turbo") == "zimage-turbo"


def test_get_config_value_returns_default_when_section_missing():
    assert get_config_value({"styles": {}}, "model", "zimage-turbo") == "zimage-turbo"


def test_get_config_value_returns_value_when_present():
    assert get_config_value(
        {"defaults": {"model": "zimage"}}, "model", "zimage-turbo"
    ) == "zimage"
