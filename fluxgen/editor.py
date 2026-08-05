import logging
import threading
from PIL import Image
from pathlib import Path
import warnings
from huggingface_hub import hf_hub_download
from typing import TYPE_CHECKING

logger = logging.getLogger("fluxgen")

# Default inference parameters for the edit pipeline
EDIT_DEFAULT_STEPS = 10
EDIT_DEFAULT_GUIDANCE = 1.0
EDIT_DEFAULT_TRUE_CFG = 4.0

# GGUF model configuration
GGUF_REPO = "unsloth/Qwen-Image-Edit-2511-GGUF"
GGUF_FILENAME = "qwen-image-edit-2511-Q4_K_M.gguf"
BASE_MODEL_CONFIG = "Qwen/Qwen-Image-Edit-2511"

# Maximum dimension (longest side) for edit outputs. Larger inputs are
# downscaled to fit while preserving aspect ratio.
#
# This is the package-level default. Callers (CLI, MCP) may override
# per-call via ``ImageEditor.edit(..., max_dimension=...)`` or by reading
# ``max_edit_dimension`` from `.fluxgen.toml`. The CLI resolver supplies
# ``max_dimension=getattr(args, "max_edit_dimension", MAX_EDIT_DIMENSION)``.
MAX_EDIT_DIMENSION = 1920


class ImageEditor:
    """Handles image editing using Qwen-Image-Edit-2511-GGUF or Flux2-Klein-Edit."""

    _cached_qwen_pipe = None
    _lock = threading.Lock()

    def __init__(self, model_name: str = "flux2-klein", quantize: int | None = None):
        self.model_name = model_name.lower()
        self.quantize = quantize
        self.device = self._get_device()
        self.pipe = None
        self.mflux_model = None

    def _get_device(self) -> str:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _get_compute_dtype(self) -> "torch.dtype":
        import torch
        if self.device in {"cuda", "mps"}:
            return torch.bfloat16
        return torch.float32

    def _load_pipeline(self):
        if self.model_name == "qwen-image-edit":
            self._load_qwen_pipeline()
        else:
            self._load_mflux_pipeline()

    def _load_qwen_pipeline(self):
        if self.pipe is not None:
            return
        with ImageEditor._lock:
            if self.pipe is not None:
                return
            if ImageEditor._cached_qwen_pipe is not None:
                self.pipe = ImageEditor._cached_qwen_pipe
                return

        from diffusers import (
            QwenImageEditPlusPipeline,
            QwenImageTransformer2DModel,
            GGUFQuantizationConfig,
        )

        compute_dtype = self._get_compute_dtype()

        logger.debug(f"Downloading/loading GGUF weights: {GGUF_REPO}/{GGUF_FILENAME}...")
        ckpt_path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILENAME)

        logger.debug(f"Loading GGUF transformer from {ckpt_path}...")
        transformer = QwenImageTransformer2DModel.from_single_file(
            ckpt_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=compute_dtype),
            config=BASE_MODEL_CONFIG,
            subfolder="transformer",
            torch_dtype=compute_dtype,
        )

        logger.info(f"Loading pipeline components ({BASE_MODEL_CONFIG}) on {self.device}...")
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            BASE_MODEL_CONFIG,
            transformer=transformer,
            torch_dtype=compute_dtype,
        )
        logger.debug("Pipeline loaded successfully.")
        if self.device == "cpu":
            self.pipe.to("cpu")
        else:
            self.pipe.enable_model_cpu_offload(device=self.device)
        self.pipe.set_progress_bar_config(disable=logger.getEffectiveLevel() > logging.INFO)
        ImageEditor._cached_qwen_pipe = self.pipe

    def _load_mflux_pipeline(self):
        if self.mflux_model is not None:
            return

        logger.info(f"Loading mflux model 'flux2-klein-edit' on MLX...")
        from fluxgen.generator import ModelManager
        self.mflux_model = ModelManager.get_model(
            model_name="flux2-klein-edit",
            quantize=self.quantize,
        )
        logger.debug("MFLUX model loaded successfully.")

    def _resolve_and_validate_inputs(
        self, image_paths: list[str]
    ) -> tuple[list[Path], tuple[int, int]]:
        """Resolve input paths, verify file integrity, and read the first image's size.

        Delegates each path's existence + ``is_file()`` + PIL integrity
        check to ``fluxgen.image_validation.validate_image_file`` (with
        ``label="input image"`` so error messages keep the pre-refactor
        "Input image not found" wording).

        The validator opens each file exactly once: it reads ``.size``
        for the first image and runs ``Image.verify()`` inside the same
        ``Image.open()`` block. The first image's file handle and pixel
        cache are invalidated after this call, so the **edit pipeline
        reopens the first file at edit() line ~219 for the RGB
        conversion** — that's expected, not a duplication bug.

        Precondition: ``image_paths`` must be non-empty. argparse's
        ``nargs="+"`` guarantees this for the CLI path; calling with
        ``[]`` raises ``ValueError`` explicitly.

        Returns:
            ``(resolved_paths, first_image_size)`` where
            ``first_image_size`` is the ``(width, height)`` of
            ``image_paths[0]``.

        Raises:
            ValueError: ``image_paths`` is empty, or a path points at a
                directory rather than a file.
            FileNotFoundError: a path does not exist.
            InvalidImageError: a file is unreadable or fails integrity
                check.
        """
        from fluxgen.image_validation import validate_image_file

        if not image_paths:
            raise ValueError("image_paths must be non-empty")

        resolved_paths: list[Path] = []
        first_size: tuple[int, int] | None = None

        for idx, path in enumerate(image_paths):
            read_size = idx == 0
            if read_size:
                resolved, size = validate_image_file(
                    path, read_size=True, label="input image"
                )
                first_size = size
            else:
                resolved = validate_image_file(path, label="input image")
            resolved_paths.append(resolved)

        # Guaranteed by the empty-list guard above + the `if idx == 0` branch.
        assert first_size is not None
        return resolved_paths, first_size

    def edit(
        self,
        image_paths: list[str],
        prompt: str,
        output_path: str,
        steps: int | None = None,
        guidance_scale: float | None = None,
        true_cfg_scale: float = EDIT_DEFAULT_TRUE_CFG,
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
        max_dimension: int = MAX_EDIT_DIMENSION,
    ) -> None:
        """Perform instruction-based image editing.

        ``max_dimension`` caps the longest side of the edit output.
        Inputs or requested dimensions larger than this are downscaled
        while preserving aspect ratio. Defaults to
        :data:`MAX_EDIT_DIMENSION` (1920 px); the CLI passes the value
        from `[defaults] max_edit_dimension` in `.fluxgen.toml` when
        present, falling back to the default.
        """
        resolved_paths, (img_w, img_h) = self._resolve_and_validate_inputs(image_paths)

        # Log warning if multiple images are provided and we are using default size from first image
        if len(resolved_paths) > 1 and (width is None or height is None):
            logger.warning(
                f"Multiple input images provided. Defaulting dimensions to the first image's size: {img_w}x{img_h}."
            )

        run_width = width if width is not None else img_w
        run_height = height if height is not None else img_h

        # Limit maximum dimensions to the configured cap (per-call or
        # package default) while preserving aspect ratio.
        if run_width > max_dimension or run_height > max_dimension:
            aspect_ratio = run_width / run_height
            if run_width > run_height:
                new_w = max_dimension
                new_h = int(round(max_dimension / aspect_ratio))
            else:
                new_h = max_dimension
                new_w = int(round(max_dimension * aspect_ratio))

            if width is not None or height is not None:
                logger.warning(
                    f"Requested dimensions ({run_width}x{run_height}) exceed the {max_dimension}px limit. "
                    f"Downscaling overrides to {new_w}x{new_h} to preserve aspect ratio."
                )
            else:
                logger.warning(
                    f"Input image dimensions ({img_w}x{img_h}) exceed the {max_dimension}px limit. "
                    f"Downscaling to {new_w}x{new_h} to preserve aspect ratio."
                )
            run_width, run_height = new_w, new_h

        self._load_pipeline()

        if self.model_name == "qwen-image-edit":
            if len(resolved_paths) > 1:
                raise ValueError("Qwen-Image-Edit only supports a single input image.")

            run_steps = steps if steps is not None else EDIT_DEFAULT_STEPS
            run_guidance = guidance_scale if guidance_scale is not None else EDIT_DEFAULT_GUIDANCE

            logger.debug(f"Opening input image: {resolved_paths[0]}")
            # verify() in the validation step invalidated the cached pixel
            # data, so we need a fresh open here for the RGB conversion.
            with Image.open(resolved_paths[0]) as raw:
                image = raw.convert("RGB")

            logger.debug(f"Editing image with prompt: '{prompt}' (size={run_width}x{run_height})...")
            import torch
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", RuntimeWarning)
                with torch.inference_mode():
                    result = self.pipe(
                        prompt=prompt,
                        negative_prompt=" ",
                        image=[image],
                        num_inference_steps=run_steps,
                        guidance_scale=run_guidance,
                        true_cfg_scale=true_cfg_scale,
                        num_images_per_prompt=1,
                        width=run_width,
                        height=run_height,
                    ).images[0]

            invalid_cast_warning = any(
                "invalid value encountered in cast" in str(warning.message)
                for warning in caught_warnings
            )
            if invalid_cast_warning or self._is_blank_black(result):
                raise RuntimeError(
                    "Image edit produced invalid/blank output. This usually means the model generated NaN values "
                    "during decode; bfloat16 is required on accelerator devices for Qwen-Image-Edit."
                )
        else:
            run_steps = steps if steps is not None else 4
            run_guidance = guidance_scale if guidance_scale is not None else 1.0

            if seed is None:
                import random
                seed = random.randint(0, 2**32 - 1)

            logger.debug(f"Editing images with prompt: '{prompt}' using flux2-klein (seed={seed}, steps={run_steps}, guidance={run_guidance}, size={run_width}x{run_height})...")
            result_wrapper = self.mflux_model.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=run_steps,
                guidance=run_guidance,
                image_paths=resolved_paths,
                width=run_width,
                height=run_height,
            )

            if hasattr(result_wrapper, "image"):
                result = result_wrapper.image
            else:
                result = result_wrapper

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(out_path)
        logger.info(f"Edited image saved to {out_path}")

    @staticmethod
    def _is_blank_black(image: Image.Image) -> bool:
        return all(channel_extrema == (0, 0) for channel_extrema in image.convert("RGB").getextrema())
