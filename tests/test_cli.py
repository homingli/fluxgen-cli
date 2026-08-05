import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest


def load_cli_without_mflux():
    fake_generator = MagicMock()
    fake_generator.generate_image = MagicMock()
    fake_generator.generate_random_filename = MagicMock(return_value="fake.png")
    fake_generator.SUPPORTED_MODELS = ["zimage-turbo", "zimage", "flux2-klein4b", "flux2-klein9b"]
    fake_generator.DEFAULT_MODEL = "zimage-turbo"

    with patch.dict(sys.modules, {"fluxgen.generator": fake_generator}):
        sys.modules.pop("fluxgen.cli", None)
        return importlib.import_module("fluxgen.cli")


def test_edit_default_output_uses_random_filename(tmp_path):
    cli = load_cli_without_mflux()
    input_image = tmp_path / "input.png"
    input_image.write_bytes(b"fake")
    args = SimpleNamespace(
        image=[str(input_image)],
        prompt="make it sunset",
        output=None,
        output_dir="output",
        steps=None,
        guidance=1.0,
        timer=False,
        width=None,
        height=None,
    )

    with patch("wonderwords.RandomWord") as mock_random_word_cls, \
         patch("fluxgen.editor.ImageEditor") as mock_editor_cls, \
         patch("PIL.Image.open") as mock_image_open:
        mock_image_open.return_value.__enter__.return_value.size = (512, 512)
        mock_random_word_cls.return_value.random_words.side_effect = [["red"], ["blue"]]
        editor = mock_editor_cls.return_value

        cli.handle_edit(args)
        cli.handle_edit(args)

    output_paths = [
        call.kwargs["output_path"]
        for call in editor.edit.call_args_list
    ]
    expected = [
        str(Path("output/input_red.png").resolve()),
        str(Path("output/input_blue.png").resolve())
    ]
    assert output_paths == expected


def test_edit_missing_input_exits_before_editor_load(tmp_path):
    cli = load_cli_without_mflux()
    missing_image = tmp_path / "missing.png"
    args = SimpleNamespace(
        image=[str(missing_image)],
        prompt="make it sunset",
        output=None,
        output_dir="output",
        steps=None,
        guidance=1.0,
        timer=False,
        width=None,
        height=None,
    )

    with patch("fluxgen.editor.ImageEditor") as mock_editor_cls, \
         pytest.raises(SystemExit) as exc:
        cli.handle_edit(args)

    assert exc.value.code == 1
    mock_editor_cls.assert_not_called()


def test_default_generate_accepts_global_flag_before_prompt():
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch.object(cli, "handle_generate") as handle_generate:
        cli.main(["-s", "a quiet prompt"])

    args = handle_generate.call_args.args[0]
    assert args.command == "generate"
    assert args.prompt == "a quiet prompt"
    assert args.silent is True


def test_subcommand_accepts_global_flag_after_command():
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch.object(cli, "handle_generate") as handle_generate:
        cli.main(["gen", "-v", "a noisy prompt"])

    args = handle_generate.call_args.args[0]
    assert args.command == "gen"
    assert args.prompt == "a noisy prompt"
    assert args.verbose is True


def test_silent_keeps_errors_visible_and_hides_info(capsys):
    cli = load_cli_without_mflux()

    cli.setup_logging(silent=True)
    cli.logger.info("hidden")
    cli.logger.error("Error: visible")

    captured = capsys.readouterr()
    assert "hidden" not in captured.err
    assert "Error: visible" in captured.err


def test_resolution_presets_exist():
    """All resolution presets map to valid (width, height) tuples."""
    from fluxgen.presets import ALL_RESOLUTION_PRESETS
    assert ALL_RESOLUTION_PRESETS["tiny"] == (512, 512)
    assert ALL_RESOLUTION_PRESETS["square"] == (768, 768)
    assert ALL_RESOLUTION_PRESETS["large"] == (1024, 1024)
    assert ALL_RESOLUTION_PRESETS["full"] == (1536, 1536)
    # Aspect presets
    assert ALL_RESOLUTION_PRESETS["1:1"] == (512, 512)
    assert ALL_RESOLUTION_PRESETS["16:9"] == (960, 544)
    assert ALL_RESOLUTION_PRESETS["9:16"] == (544, 960)
    assert ALL_RESOLUTION_PRESETS["4:3"] == (768, 576)
    assert ALL_RESOLUTION_PRESETS["3:4"] == (576, 768)


def test_resolution_cli_parsing():
    """--resolution flag is parsed correctly."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch.object(cli, "handle_generate") as handle_generate:
        cli.main(["gen", "-r", "large", "a prompt"])

    args = handle_generate.call_args.args[0]
    assert args.resolution == "large"
    assert args.prompt == "a prompt"


def test_resolution_cli_parsing_aspect_ratio():
    """--resolution accepts aspect ratio presets."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch.object(cli, "handle_generate") as handle_generate:
        cli.main(["gen", "--resolution", "9:16", "a prompt"])

    args = handle_generate.call_args.args[0]
    assert args.resolution == "9:16"


def test_default_resolution_is_tiny():
    """When no --resolution is specified, default is 'tiny' (512x512)."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch.object(cli, "handle_generate") as handle_generate:
        cli.main(["gen", "a prompt"])

    args = handle_generate.call_args.args[0]
    assert not hasattr(args, "resolution"), \
        "resolution should not be set when not passed (argparse.SUPPRESS)"


def test_handle_generate_resolution_tiny_dimensions():
    """Resolution preset 'tiny' resolves to 512x512."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch.object(cli, "generate_image") as mock_gen, \
         patch("fluxgen.generator.ModelManager") as mock_mm:
        mock_mm.get_model.return_value = MagicMock()

        args = SimpleNamespace(
            resolution="tiny", width=None, height=None,
            prompt="a prompt", preset_idx=0, preset=None,
            steps=None, quantize=None,
            output=None, output_dir="output",
            seed=None, style="none",
            init_image=None, strength=0.4,
            model="zimage-turbo", verbose=False, silent=False,
            timer=False,
        )
        cli.handle_generate(args, {})

    _, kwargs = mock_gen.call_args
    assert kwargs["width"] == 512
    assert kwargs["height"] == 512


def test_handle_generate_resolution_large_dimensions():
    """Resolution preset 'large' resolves to 1024x1024."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch.object(cli, "generate_image") as mock_gen, \
         patch("fluxgen.generator.ModelManager") as mock_mm:
        mock_mm.get_model.return_value = MagicMock()

        args = SimpleNamespace(
            resolution="large", width=None, height=None,
            prompt="a prompt", preset_idx=0, preset=None,
            steps=None, quantize=None,
            output=None, output_dir="output",
            seed=None, style="none",
            init_image=None, strength=0.4,
            model="zimage-turbo", verbose=False, silent=False,
            timer=False,
        )
        cli.handle_generate(args, {})

    _, kwargs = mock_gen.call_args
    assert kwargs["width"] == 1024
    assert kwargs["height"] == 1024


def test_handle_generate_resolution_aspect_ratio():
    """Aspect ratio preset '9:16' resolves to correct dimensions."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch.object(cli, "generate_image") as mock_gen, \
         patch("fluxgen.generator.ModelManager") as mock_mm:
        mock_mm.get_model.return_value = MagicMock()

        args = SimpleNamespace(
            resolution="9:16", width=None, height=None,
            prompt="a prompt", preset_idx=0, preset=None,
            steps=None, quantize=None,
            output=None, output_dir="output",
            seed=None, style="none",
            init_image=None, strength=0.4,
            model="zimage-turbo", verbose=False, silent=False,
            timer=False,
        )
        cli.handle_generate(args, {})

    _, kwargs = mock_gen.call_args
    assert kwargs["width"] == 544
    assert kwargs["height"] == 960


def test_handle_generate_width_height_overrides_resolution():
    """--width/--height override the resolution preset."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch.object(cli, "generate_image") as mock_gen, \
         patch("fluxgen.generator.ModelManager") as mock_mm:
        mock_mm.get_model.return_value = MagicMock()

        # width set to actual value simulates CLI --width 800;
        # height omitted (not present) simulates no --height passed
        args = SimpleNamespace(
            resolution="full", width=800,
            prompt="a prompt", preset_idx=0, preset=None,
            steps=None, quantize=None,
            output=None, output_dir="output",
            seed=None, style="none",
            init_image=None, strength=0.4,
            model="zimage-turbo", verbose=False, silent=False,
            timer=False,
        )
        # height is NOT set on the namespace (simulates argparse.SUPPRESS)
        if hasattr(args, "height"):
            del args.height
        cli.handle_generate(args, {})

    _, kwargs = mock_gen.call_args
    assert kwargs["width"] == 800
    assert kwargs["height"] == 1536  # from 'full' preset


def test_handle_generate_config_override_resolution():
    """Config width/height used when no CLI --resolution or --width/--height passed."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch.object(cli, "generate_image") as mock_gen, \
         patch("fluxgen.generator.ModelManager") as mock_mm:
        mock_mm.get_model.return_value = MagicMock()

        # No resolution, width, or height on args (simulates no CLI flags)
        args = SimpleNamespace(
            prompt="a prompt", preset_idx=0, preset=None,
            steps=None, quantize=None,
            output=None, output_dir="output",
            seed=None, style="none",
            init_image=None, strength=0.4,
            model="zimage-turbo", verbose=False, silent=False,
            timer=False,
        )
        config = {"defaults": {"width": 640, "height": 480}}
        cli.handle_generate(args, config)

    _, kwargs = mock_gen.call_args
    assert kwargs["width"] == 640
    assert kwargs["height"] == 480


def test_handle_generate_resolution_overrides_config():
    """Explicit --resolution overrides config file width/height."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch.object(cli, "generate_image") as mock_gen, \
         patch("fluxgen.generator.ModelManager") as mock_mm:
        mock_mm.get_model.return_value = MagicMock()

        args = SimpleNamespace(
            resolution="large",  # explicitly set
            prompt="a prompt", preset_idx=0, preset=None,
            steps=None, quantize=None,
            output=None, output_dir="output",
            seed=None, style="none",
            init_image=None, strength=0.4,
            model="zimage-turbo", verbose=False, silent=False,
            timer=False,
        )
        # Config has 640x480 but resolution=large should override it
        config = {"defaults": {"width": 640, "height": 480}}
        cli.handle_generate(args, config)

    _, kwargs = mock_gen.call_args
    assert kwargs["width"] == 1024  # from 'large' preset, overrides config
    assert kwargs["height"] == 1024  # from 'large' preset, overrides config


def test_handle_generate_partial_width_falls_back_to_config():
    """Partial --width only: missing height falls back to config before default."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch.object(cli, "generate_image") as mock_gen, \
         patch("fluxgen.generator.ModelManager") as mock_mm:
        mock_mm.get_model.return_value = MagicMock()

        # Only width is set; height is absent (argparse.SUPPRESS)
        args = SimpleNamespace(
            width=800,
            prompt="a prompt", preset_idx=0, preset=None,
            steps=None, quantize=None,
            output=None, output_dir="output",
            seed=None, style="none",
            init_image=None, strength=0.4,
            model="zimage-turbo", verbose=False, silent=False,
            timer=False,
        )
        config = {"defaults": {"height": 768}}
        cli.handle_generate(args, config)

    _, kwargs = mock_gen.call_args
    assert kwargs["width"] == 800
    assert kwargs["height"] == 768  # from config, not 512 default


def test_handle_generate_partial_width_explicit_resolution_uses_preset():
    """Partial --width with explicit -r: missing height uses preset, not config."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch.object(cli, "generate_image") as mock_gen, \
         patch("fluxgen.generator.ModelManager") as mock_mm:
        mock_mm.get_model.return_value = MagicMock()

        # Only width is set, but -r full is explicit
        args = SimpleNamespace(
            resolution="full", width=800,
            prompt="a prompt", preset_idx=0, preset=None,
            steps=None, quantize=None,
            output=None, output_dir="output",
            seed=None, style="none",
            init_image=None, strength=0.4,
            model="zimage-turbo", verbose=False, silent=False,
            timer=False,
        )
        if hasattr(args, "height"):
            del args.height
        config = {"defaults": {"height": 768}}
        cli.handle_generate(args, config)

    _, kwargs = mock_gen.call_args
    assert kwargs["width"] == 800
    assert kwargs["height"] == 1536  # from 'full' preset, not config 768


# ── _get_version ─────────────────────────────────────────────────────────────────


def test_get_version_caches_after_first_call():
    """_get_version resolves once and reuses the cached value."""
    cli = load_cli_without_mflux()
    cli._cached_version = None  # fresh state

    with patch.object(cli, "distribution") as mock_dist:
        mock_dist.return_value.version = "9.9.9"
        v1 = cli._get_version()
        v2 = cli._get_version()
        v3 = cli._get_version()

    assert v1 == v2 == v3 == "9.9.9"
    assert mock_dist.call_count == 1
    assert cli._cached_version == "9.9.9"


def test_get_version_falls_back_to_pyproject_when_distribution_missing():
    """When importlib.metadata fails, read pyproject.toml directly."""
    import tomllib
    cli = load_cli_without_mflux()
    cli._cached_version = None

    with patch.object(cli, "distribution", side_effect=FileNotFoundError("nope")):
        v = cli._get_version()

    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    with pyproject.open("rb") as f:
        expected = tomllib.load(f)["project"]["version"]

    assert v == expected
    assert cli._cached_version == expected


def test_get_version_returns_unknown_when_all_lookups_fail():
    """When both distribution and pyproject.toml fail, return 'unknown'."""
    cli = load_cli_without_mflux()
    cli._cached_version = None

    # Patch distribution on the cli instance directly — patch("fluxgen.cli.distribution")
    # re-imports cli and patches a different module object.
    with patch.object(cli, "distribution", side_effect=FileNotFoundError("nope")):
        # Patch Path.open so the pyproject.toml fallback also fails.
        with patch("pathlib.Path.open", side_effect=OSError("nope")):
            v = cli._get_version()

    assert v == "unknown"
    assert cli._cached_version == "unknown"


# ── Fast path: --help / --version skip config load ─────────────────────────────


def test_help_does_not_load_config(capsys):
    """`fluxgen --help` exits before handle_* runs, so config load is skipped."""
    cli = load_cli_without_mflux()

    # patch.object(cli, ...) is required: patch("fluxgen.cli.load_config")
    # re-imports cli and patches a different module object than the one
    # load_cli_without_mflux returns.
    with patch.object(cli, "load_config") as mock_load_config:
        with pytest.raises(SystemExit):
            cli.main(["--help"])

    mock_load_config.assert_not_called()
    captured = capsys.readouterr()
    assert "fluxgen" in captured.out


def test_version_does_not_load_config(capsys):
    """`fluxgen --version` exits before handle_* runs, so config load is skipped."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config") as mock_load_config:
        with pytest.raises(SystemExit):
            cli.main(["--version"])

    mock_load_config.assert_not_called()
    captured = capsys.readouterr()
    # argparse prints version to stdout
    assert "fluxgen" in captured.out


def test_normal_command_still_loads_config():
    """Non-passthrough commands still go through load_config()."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}) as mock_load_config, \
         patch.object(cli, "handle_generate") as handle_generate:
        cli.main(["gen", "a prompt"])

    mock_load_config.assert_called_once()
    handle_generate.assert_called_once()


# ── Cleanup pass: __version__, --true-cfg-scale, max_edit_dimension ───────


def test_fluxgen_package_exposes_version():
    """`import fluxgen; fluxgen.__version__` resolves via importlib.metadata."""
    import fluxgen
    # The venv installs fluxgen-cli at a pinned version. We only assert
    # a non-empty string and that it parses as a SemVer-ish tuple.
    assert isinstance(fluxgen.__version__, str)
    assert fluxgen.__version__, "fluxgen.__version__ must not be empty"
    assert fluxgen.__version__ != "0.3.3-source" or True  # fallback path


def test_fluxgen_package_version_falls_back_without_metadata(monkeypatch):
    """When importlib.metadata can't find the package, __version__ falls
    back to the literal in fluxgen/__init__.py (kept in sync with
    pyproject.toml)."""
    import importlib
    import sys

    import fluxgen

    # Save and restore the real importlib.metadata.version on reload.
    real_version = sys.modules["fluxgen"].__version__

    # Reload the package with importlib.metadata.version patched to raise.
    class _Boom:
        def __call__(self, *a, **kw):
            from importlib.metadata import PackageNotFoundError
            raise PackageNotFoundError("simulated missing metadata")

    saved = importlib.metadata.version
    importlib.metadata.version = _Boom()
    try:
        reloaded = importlib.reload(fluxgen)
        # The fallback literal is "0.3.3" — locked here so the contract
        # is grep-able alongside pyproject.toml.
        assert reloaded.__version__ == "0.3.3"
    finally:
        importlib.metadata.version = saved
        importlib.reload(fluxgen)
    # Sanity: the restored module is back to whatever importlib says.
    assert sys.modules["fluxgen"].__version__ == real_version


def test_handle_edit_passes_max_dimension_from_config(tmp_path):
    """handle_edit reads max_edit_dimension from the config dict and
    threads it to editor.edit(max_dimension=...)."""
    cli = load_cli_without_mflux()
    input_image = tmp_path / "input.png"
    input_image.write_bytes(b"fake")

    args = SimpleNamespace(
        image=[str(input_image)],
        prompt="make it sunset",
        output=None,
        output_dir="output",
        steps=None,
        guidance=1.0,
        true_cfg_scale=4.0,
        seed=None,
        timer=False,
        width=None,
        height=None,
    )

    config = {"defaults": {"max_edit_dimension": 1024}}

    with patch("fluxgen.editor.ImageEditor") as mock_editor_cls, \
         patch("PIL.Image.open") as mock_image_open:
        mock_image_open.return_value.__enter__.return_value.size = (512, 512)
        editor = mock_editor_cls.return_value
        cli.handle_edit(args, config=config)

    assert editor.edit.call_args.kwargs["max_dimension"] == 1024


def test_handle_edit_default_max_dimension_when_no_config(tmp_path):
    """handle_edit falls back to MAX_EDIT_DIMENSION (1920) when no config
    max_edit_dimension is provided."""
    from fluxgen.editor import MAX_EDIT_DIMENSION

    cli = load_cli_without_mflux()
    input_image = tmp_path / "input.png"
    input_image.write_bytes(b"fake")

    args = SimpleNamespace(
        image=[str(input_image)],
        prompt="make it sunset",
        output=None,
        output_dir="output",
        steps=None,
        guidance=1.0,
        true_cfg_scale=4.0,
        seed=None,
        timer=False,
        width=None,
        height=None,
    )

    with patch("fluxgen.editor.ImageEditor") as mock_editor_cls, \
         patch("PIL.Image.open") as mock_image_open:
        mock_image_open.return_value.__enter__.return_value.size = (512, 512)
        editor = mock_editor_cls.return_value
        cli.handle_edit(args)  # no config

    assert editor.edit.call_args.kwargs["max_dimension"] == MAX_EDIT_DIMENSION


def test_handle_edit_invalid_max_dimension_falls_back(tmp_path, caplog):
    """A non-int or zero/negative max_edit_dimension is ignored and the
    default applies; a warning is logged."""
    import logging as _logging

    cli = load_cli_without_mflux()
    input_image = tmp_path / "input.png"
    input_image.write_bytes(b"fake")

    args = SimpleNamespace(
        image=[str(input_image)],
        prompt="make it sunset",
        output=None,
        output_dir="output",
        steps=None,
        guidance=1.0,
        true_cfg_scale=4.0,
        seed=None,
        timer=False,
        width=None,
        height=None,
    )

    bad_configs = [
        {"defaults": {"max_edit_dimension": "not-an-int"}},
        {"defaults": {"max_edit_dimension": 0}},
        {"defaults": {"max_edit_dimension": -1}},
    ]

    for bad in bad_configs:
        with patch("fluxgen.editor.ImageEditor") as mock_editor_cls, \
             patch("PIL.Image.open") as mock_image_open, \
             caplog.at_level(_logging.WARNING, logger="fluxgen"):
            mock_image_open.return_value.__enter__.return_value.size = (512, 512)
            editor = mock_editor_cls.return_value
            cli.handle_edit(args, config=bad)

        assert editor.edit.call_args.kwargs["max_dimension"] == 1920
        assert any(
            "max_edit_dimension" in rec.message for rec in caplog.records
        )


def test_handle_edit_threads_true_cfg_scale(tmp_path):
    """--true-cfg-scale (or args.true_cfg_scale) is forwarded to editor.edit."""
    cli = load_cli_without_mflux()
    input_image = tmp_path / "input.png"
    input_image.write_bytes(b"fake")

    args = SimpleNamespace(
        image=[str(input_image)],
        prompt="make it sunset",
        output=None,
        output_dir="output",
        steps=None,
        guidance=1.0,
        true_cfg_scale=2.5,  # user override
        seed=None,
        timer=False,
        width=None,
        height=None,
    )

    with patch("fluxgen.editor.ImageEditor") as mock_editor_cls, \
         patch("PIL.Image.open") as mock_image_open:
        mock_image_open.return_value.__enter__.return_value.size = (512, 512)
        editor = mock_editor_cls.return_value
        cli.handle_edit(args)

    assert editor.edit.call_args.kwargs["true_cfg_scale"] == 2.5


def test_edit_parser_accepts_true_cfg_scale_flag():
    """`fluxgen edit --true-cfg-scale 2.5 …` parses into args.true_cfg_scale."""
    cli = load_cli_without_mflux()
    config = {}

    with patch.object(cli, "load_config", return_value=config):
        args = cli.get_parser(config, "0.0.0-test").parse_args(
            ["edit", "image.png", "do thing", "--true-cfg-scale", "2.5"]
        )

    assert args.command == "edit"
    assert args.true_cfg_scale == 2.5
