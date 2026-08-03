import pytest
import torch
from unittest.mock import ANY, MagicMock, patch
from pathlib import Path
from fluxgen.editor import ImageEditor, EDIT_DEFAULT_STEPS, EDIT_DEFAULT_GUIDANCE, EDIT_DEFAULT_TRUE_CFG
@pytest.fixture(autouse=True)
def reset_editor_cache():
    ImageEditor._cached_qwen_pipe = None
    from fluxgen.generator import ModelManager
    ModelManager.reset()
    yield
    ImageEditor._cached_qwen_pipe = None
    ModelManager.reset()



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
    assert EDIT_DEFAULT_STEPS == 10
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
    mock_input_image.size = (512, 512)
    mock_image_open.return_value.__enter__.return_value = mock_input_image
    mock_image_open.return_value.convert.return_value = mock_input_image

    # Initialize editor and force CPU
    editor = ImageEditor(model_name="qwen-image-edit")
    editor.device = "cpu"

    # Run edit with explicit steps
    editor.edit(
        image_paths=[str(input_image)],
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

    editor = ImageEditor(model_name="qwen-image-edit")
    editor.pipe = MagicMock()

    mock_input_image = MagicMock()
    mock_input_image.size = (512, 512)
    mock_image_open.return_value.__enter__.return_value = mock_input_image
    mock_image_open.return_value.convert.return_value = mock_input_image

    mock_output_image = MagicMock()
    mock_output_image.convert.return_value.getextrema.return_value = ((0, 0), (0, 0), (0, 0))
    editor.pipe.return_value.images = [mock_output_image]

    with pytest.raises(RuntimeError, match="invalid/blank output"):
        editor.edit(
            image_paths=[str(input_image)],
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
    mock_input_image = MagicMock()
    mock_input_image.size = (512, 512)
    mock_image_open.return_value.__enter__.return_value = mock_input_image
    mock_image_open.return_value.convert.return_value = mock_input_image

    editor = ImageEditor(model_name="qwen-image-edit")
    editor.device = "cpu"

    # Call without specifying steps — should use EDIT_DEFAULT_STEPS
    editor.edit(
        image_paths=[str(input_image)],
        prompt="test",
        output_path="output/out.png",
    )

    infer_kwargs = mock_pipeline.call_args.kwargs
    assert infer_kwargs["num_inference_steps"] == EDIT_DEFAULT_STEPS


def test_editor_missing_input_fails_before_pipeline_load(tmp_path):
    editor = ImageEditor(model_name="qwen-image-edit")
    editor._load_pipeline = MagicMock()
    missing_image = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError, match="Input image not found"):
        editor.edit(
            image_paths=[str(missing_image)],
            prompt="test",
            output_path="output/out.png",
        )

    editor._load_pipeline.assert_not_called()


# ── _resolve_and_validate_inputs ──────────────────────────────────────────────────────


def test_resolve_and_validate_inputs_returns_first_image_size(tmp_path):
    """Helper returns the (width, height) of the first image."""
    from PIL import Image as PILImage

    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    PILImage.new("RGB", (640, 480), "red").save(image_a)
    PILImage.new("RGB", (1024, 768), "blue").save(image_b)

    editor = ImageEditor(model_name="flux2-klein")
    paths, size = editor._resolve_and_validate_inputs([str(image_a), str(image_b)])

    assert len(paths) == 2
    assert paths[0] == image_a.resolve()
    assert paths[1] == image_b.resolve()
    assert size == (640, 480)


def test_resolve_and_validate_inputs_rejects_missing_file(tmp_path):
    editor = ImageEditor(model_name="flux2-klein")
    missing = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError, match="Input image not found"):
        editor._resolve_and_validate_inputs([str(missing)])


def test_resolve_and_validate_inputs_rejects_empty_list():
    """Empty image_paths raises ValueError (no fallback to defaults)."""
    editor = ImageEditor(model_name="flux2-klein")

    with pytest.raises(ValueError, match="must be non-empty"):
        editor._resolve_and_validate_inputs([])


def test_resolve_and_validate_inputs_rejects_directory(tmp_path):
    editor = ImageEditor(model_name="flux2-klein")
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    with pytest.raises(ValueError, match="must be a file"):
        editor._resolve_and_validate_inputs([str(subdir)])


def test_resolve_and_validate_inputs_rejects_corrupt_image(tmp_path):
    from fluxgen.exceptions import InvalidImageError

    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not a valid image file at all")

    editor = ImageEditor(model_name="flux2-klein")
    with pytest.raises(InvalidImageError):
        editor._resolve_and_validate_inputs([str(corrupt)])


def test_resolve_and_validate_inputs_size_in_same_open_as_verify(tmp_path):
    """First image is opened once for both size lookup and verify.

    This is the I/O optimization: previously the helper opened the first
    image twice (once for verify, once for size). Now both happen in one
    open (.size is a free header lookup before verify() walks the file).
    """
    from PIL import Image as PILImage

    image = tmp_path / "a.png"
    PILImage.new("RGB", (100, 200), "red").save(image)

    editor = ImageEditor(model_name="flux2-klein")

    real_open = PILImage.open
    open_count = 0

    def counting_open(path, *args, **kwargs):
        nonlocal open_count
        open_count += 1
        return real_open(path, *args, **kwargs)

    with patch("PIL.Image.open", new=counting_open):
        paths, size = editor._resolve_and_validate_inputs([str(image)])

    assert open_count == 1
    assert size == (100, 200)


def test_resolve_and_validate_inputs_multi_image_one_open_per_file(tmp_path):
    """Multi-image: one open per file (verify only for non-first images)."""
    from PIL import Image as PILImage

    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    image_c = tmp_path / "c.png"
    PILImage.new("RGB", (640, 480), "red").save(image_a)
    PILImage.new("RGB", (1024, 768), "blue").save(image_b)
    PILImage.new("RGB", (320, 240), "green").save(image_c)

    editor = ImageEditor(model_name="flux2-klein")

    real_open = PILImage.open
    open_count = 0

    def counting_open(path, *args, **kwargs):
        nonlocal open_count
        open_count += 1
        return real_open(path, *args, **kwargs)

    with patch("PIL.Image.open", new=counting_open):
        paths, size = editor._resolve_and_validate_inputs(
            [str(image_a), str(image_b), str(image_c)]
        )

    assert open_count == 3  # one open per image, no extra size re-read
    assert size == (640, 480)


def test_resolve_and_validate_inputs_expands_user_and_resolves(tmp_path):
    """Paths with ~ and relative components are resolved."""
    from PIL import Image as PILImage

    image = tmp_path / "a.png"
    PILImage.new("RGB", (50, 50), "red").save(image)

    editor = ImageEditor(model_name="flux2-klein")
    paths, _ = editor._resolve_and_validate_inputs([str(image)])

    assert paths[0].is_absolute()
    assert paths[0] == image.resolve()

