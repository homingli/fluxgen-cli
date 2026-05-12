import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def load_cli_without_mflux():
    fake_generator = MagicMock()
    fake_generator.generate_image = MagicMock()
    fake_generator.generate_random_filename = MagicMock(return_value="fake.png")
    fake_generator.SUPPORTED_MODELS = ["zimage-turbo", "zimage", "flux1-schnell"]
    fake_generator.DEFAULT_MODEL = "zimage-turbo"

    with patch.dict(sys.modules, {"fluxgen.generator": fake_generator}):
        sys.modules.pop("fluxgen.cli", None)
        return importlib.import_module("fluxgen.cli")


def test_edit_default_output_uses_random_filename():
    cli = load_cli_without_mflux()
    args = SimpleNamespace(
        image="input.png",
        prompt="make it sunset",
        output=None,
        output_dir="output",
        steps=None,
        guidance=1.0,
        timer=False,
    )

    with patch.object(cli, "generate_random_filename", side_effect=["one-two-red.png", "four-five-blue.png"]), \
         patch("fluxgen.editor.ImageEditor") as mock_editor_cls:
        editor = mock_editor_cls.return_value

        cli.handle_edit(args)
        cli.handle_edit(args)

    output_paths = [
        call.kwargs["output_path"]
        for call in editor.edit.call_args_list
    ]
    assert output_paths == ["output/one-two-red.png", "output/four-five-blue.png"]
