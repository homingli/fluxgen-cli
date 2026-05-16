import pytest
import torch
from unittest.mock import ANY, MagicMock, patch
from pathlib import Path
from fluxgen.editor import ImageEditor, EDIT_DEFAULT_STEPS, EDIT_DEFAULT_GUIDANCE, EDIT_DEFAULT_TRUE_CFG


def test_editor_device_selection():
    with patch("torch.backends.mps.is_available", return_value=True):
        editor = ImageEditor()
        assert editor._get_device() == "mps"

    with patch("torch.backends.mps.is_available", return_value=False), \
         patch("torch.cuda.is_available", return_value=True):
        editor = ImageEditor()
        assert editor._get_device() == "cuda"

    with patch("torch.backends.mps.is_available", return_value=False), \
         patch("torch.cuda.is_available", return_value=False):
        editor = ImageEditor()
        assert editor._get_device() == "cpu"


def test_editor_defaults():
    """Verify the module-level defaults are sane."""
    assert EDIT_DEFAULT_STEPS == 40
    assert EDIT_DEFAULT_GUIDANCE == 1.0
    assert EDIT_DEFAULT_TRUE_CFG == 4.0


def test_editor_compute_dtype_uses_bfloat16_on_accelerators():
    editor = ImageEditor()

    editor.device = "mps"
    assert editor._get_compute_dtype() == torch.bfloat16

    editor.device = "cuda"
    assert editor._get_compute_dtype() == torch.bfloat16

    editor.device = "cpu"
    assert editor._get_compute_dtype() == torch.float32


def test_editor_init_no_pipeline_loaded():
    """Pipeline should not be loaded on construction."""
    editor = ImageEditor()
    assert editor.pipe is None


@patch("fluxgen.editor.hf_hub_download")
@patch("diffusers.QwenImageEditPlusPipeline")
@patch("diffusers.QwenImageTransformer2DModel")
@patch("diffusers.GGUFQuantizationConfig")
@patch("PIL.Image.open")
def test_editor_edit_flow(mock_image_open, mock_gguf_config, mock_transformer_cls, mock_pipeline_cls, mock_hf_download, tmp_path):
    """Verify the full edit flow wires up correctly."""
    input_image = tmp_path / "dummy.png"
    input_image.write_bytes(b"fake")

    # Setup mocks
    mock_hf_download.return_value = "/tmp/dummy.gguf"
    mock_transformer = MagicMock()
    mock_transformer_cls.from_single_file.return_value = mock_transformer

    mock_pipeline = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = mock_pipeline

    mock_output_image = MagicMock()
    mock_output_image.convert.return_value.getextrema.return_value = ((0, 255), (0, 255), (0, 255))
    mock_pipeline.return_value.images = [mock_output_image]

    mock_input_image = MagicMock()
    mock_image_open.return_value.convert.return_value = mock_input_image

    # Initialize editor and force CPU
    editor = ImageEditor()
    editor.device = "cpu"

    # Run edit with explicit steps
    editor.edit(
        image_path=str(input_image),
        prompt="make it red",
        output_path="output/edited.png",
        steps=10,
    )

    # Verify transformer was loaded from local path (after download)
    mock_hf_download.assert_called_once()
    mock_transformer_cls.from_single_file.assert_called_once_with(
        "/tmp/dummy.gguf",
        quantization_config=mock_gguf_config.return_value,
        config="Qwen/Qwen-Image-Edit-2511",
        subfolder="transformer",
        torch_dtype=ANY,
    )

    # Verify output was saved
    mock_output_image.save.assert_called_once()


@patch("PIL.Image.open")
def test_editor_rejects_blank_black_output(mock_image_open, tmp_path):
    input_image = tmp_path / "dummy.png"
    input_image.write_bytes(b"fake")

    editor = ImageEditor()
    editor.pipe = MagicMock()

    mock_input_image = MagicMock()
    mock_image_open.return_value.convert.return_value = mock_input_image

    mock_output_image = MagicMock()
    mock_output_image.convert.return_value.getextrema.return_value = ((0, 0), (0, 0), (0, 0))
    editor.pipe.return_value.images = [mock_output_image]

    with pytest.raises(RuntimeError, match="invalid/blank output"):
        editor.edit(
            image_path=str(input_image),
            prompt="test",
            output_path="output/out.png",
        )

    mock_output_image.save.assert_not_called()


@patch("fluxgen.editor.hf_hub_download")
@patch("diffusers.QwenImageEditPlusPipeline")
@patch("diffusers.QwenImageTransformer2DModel")
@patch("diffusers.GGUFQuantizationConfig")
@patch("PIL.Image.open")
def test_editor_uses_default_steps(mock_image_open, mock_gguf_config, mock_transformer_cls, mock_pipeline_cls, mock_hf_download, tmp_path):
    """When no steps are provided, the default (40) should be used."""
    input_image = tmp_path / "dummy.png"
    input_image.write_bytes(b"fake")

    mock_hf_download.return_value = "/tmp/dummy.gguf"
    mock_transformer_cls.from_single_file.return_value = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
    mock_output_image = MagicMock()
    mock_output_image.convert.return_value.getextrema.return_value = ((0, 255), (0, 255), (0, 255))
    mock_pipeline.return_value.images = [mock_output_image]
    mock_image_open.return_value.convert.return_value = MagicMock()

    editor = ImageEditor()
    editor.device = "cpu"

    # Call without specifying steps — should use EDIT_DEFAULT_STEPS
    editor.edit(
        image_path=str(input_image),
        prompt="test",
        output_path="output/out.png",
    )

    infer_kwargs = mock_pipeline.call_args.kwargs
    assert infer_kwargs["num_inference_steps"] == EDIT_DEFAULT_STEPS


def test_editor_missing_input_fails_before_pipeline_load(tmp_path):
    editor = ImageEditor()
    editor._load_pipeline = MagicMock()
    missing_image = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError, match="Input image not found"):
        editor.edit(
            image_path=str(missing_image),
            prompt="test",
            output_path="output/out.png",
        )

    editor._load_pipeline.assert_not_called()
