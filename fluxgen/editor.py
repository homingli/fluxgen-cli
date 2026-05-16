import logging
import torch
from PIL import Image
from pathlib import Path
import warnings
from huggingface_hub import hf_hub_download

logger = logging.getLogger("fluxgen")

# Default inference parameters for the edit pipeline
EDIT_DEFAULT_STEPS = 10
EDIT_DEFAULT_GUIDANCE = 1.0
EDIT_DEFAULT_TRUE_CFG = 4.0

# GGUF model configuration
GGUF_REPO = "unsloth/Qwen-Image-Edit-2511-GGUF"
GGUF_FILENAME = "qwen-image-edit-2511-Q4_K_M.gguf"
BASE_MODEL_CONFIG = "Qwen/Qwen-Image-Edit-2511"


class ImageEditor:
    """Handles image editing using the Qwen-Image-Edit-2511 model via Diffusers.

    Uses the unsloth Q4_K_M GGUF quantization (~13GB) for significantly
    reduced memory usage compared to the full-precision model (~58GB).
    """

    def __init__(self):
        self.device = self._get_device()
        self.pipe = None

    def _get_device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _get_compute_dtype(self) -> torch.dtype:
        # Qwen-Image-Edit is trained/recommended for bfloat16. float16 on MPS
        # can overflow during denoising/VAE decode and produce black NaN images.
        if self.device in {"cuda", "mps"}:
            return torch.bfloat16
        return torch.float32

    def _load_pipeline(self):
        if self.pipe is not None:
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

    def edit(
        self,
        image_path: str,
        prompt: str,
        output_path: str,
        steps: int = EDIT_DEFAULT_STEPS,
        guidance_scale: float = EDIT_DEFAULT_GUIDANCE,
        true_cfg_scale: float = EDIT_DEFAULT_TRUE_CFG,
    ) -> None:
        """Perform instruction-based image editing."""
        input_path = Path(image_path).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")
        if not input_path.is_file():
            raise ValueError(f"Input image must be a file: {input_path}")

        self._load_pipeline()

        logger.debug(f"Opening input image: {input_path}")
        image = Image.open(input_path).convert("RGB")

        logger.debug(f"Editing image with prompt: '{prompt}'...")
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", RuntimeWarning)
            with torch.inference_mode():
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=" ",
                    image=[image],
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    true_cfg_scale=true_cfg_scale,
                    num_images_per_prompt=1,
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

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(out_path)
        logger.info(f"Edited image saved to {out_path}")

    @staticmethod
    def _is_blank_black(image: Image.Image) -> bool:
        return all(channel_extrema == (0, 0) for channel_extrema in image.convert("RGB").getextrema())
