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
