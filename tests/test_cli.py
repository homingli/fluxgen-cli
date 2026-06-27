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
