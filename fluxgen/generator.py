import logging
import random
from pathlib import Path
from PIL import Image

from mflux.models.common.config import ModelConfig

# Supported model identifiers
SUPPORTED_MODELS = ["zimage-turbo", "zimage", "flux2-klein4b", "flux2-klein9b"]
DEFAULT_MODEL = "zimage-turbo"

logger = logging.getLogger("fluxgen")


class ModelManager:
    """Manages model instances with caching and multi-model support.

    Supported models:
      - zimage-turbo   (default) — fast, guidance-free ZImage variant
      - zimage                 — full ZImage with guidance support
      - flux2-klein4b          — FLUX.2 Klein 4B model (default)
      - flux2-klein9b          — FLUX.2 Klein 9B model
    """

    _instance = None
    _current_config = None

    @classmethod
    def get_model(cls, model_name: str, quantize: int | None = None):
        """Return a cached model instance, re-creating only when config changes."""
        model_name = model_name.lower()
        if model_name not in SUPPORTED_MODELS and model_name != "flux2-klein-edit":
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Choose from: {', '.join(SUPPORTED_MODELS)}"
            )

        config_key = (model_name, quantize)
        if cls._instance is None or cls._current_config != config_key:
            cls._instance = cls._create_model(model_name, quantize)
            cls._current_config = config_key
        return cls._instance

    @classmethod
    def _create_model(cls, model_name: str, quantize: int | None):
        """Instantiate the appropriate model class."""
        if model_name == "zimage":
            from mflux.models.z_image import ZImage

            return ZImage(
                quantize=quantize,
                model_config=ModelConfig.z_image(),
            )
        elif model_name == "flux2-klein4b":
            from mflux.models.flux2.variants import Flux2Klein

            return Flux2Klein(
                quantize=quantize,
                model_config=ModelConfig.flux2_klein_4b(),
            )
        elif model_name == "flux2-klein9b":
            from mflux.models.flux2.variants import Flux2Klein

            return Flux2Klein(
                quantize=quantize,
                model_config=ModelConfig.flux2_klein_9b(),
            )
        elif model_name == "flux2-klein-edit":
            from mflux.models.flux2.variants import Flux2KleinEdit

            return Flux2KleinEdit(
                quantize=quantize,
                model_config=ModelConfig.flux2_klein_9b(),
            )
        else:  # zimage-turbo (default)
            from mflux.models.z_image import ZImageTurbo

            return ZImageTurbo(
                quantize=quantize,
                model_config=ModelConfig.z_image_turbo(),
            )

    @classmethod
    def reset(cls):
        """Clear the cached model (useful for switching models)."""
        cls._instance = None
        cls._current_config = None


# ── Model-specific default parameters ────────────────────────────────────────

MODEL_DEFAULTS = {
    "zimage-turbo": {
        "guidance": 0.0,       # turbo ignores guidance
        "steps": 4,
    },
    "zimage": {
        "guidance": 4.0,       # supports classifier-free guidance
        "steps": 20,
    },
    "flux2-klein4b": {
        "guidance": 3.5,
        "steps": 4,
    },
    "flux2-klein9b": {
        "guidance": 3.5,
        "steps": 4,
    },
}


from fluxgen.styling import StyleManager

def generate_random_filename() -> str:
    """Generate a random 3-word filename with .png extension"""
    try:
        from wonderwords import RandomWord
        rw = RandomWord()
        words = rw.random_words(3, word_max_length=5)
        return "-".join(words) + ".png"
    except Exception:
        import time
        return f"generated-{int(time.time())}.png"

def generate_image(
    prompt: str,
    preset: dict,
    seed: int | None = None,
    output: str = "output.png",
    width: int = 512,
    height: int = 512,
    style: str = "ghibli",
    custom_styles: dict[str, str] | None = None,
    init_image: str | None = None,
    strength: float = 0.4,
    model_name: str = DEFAULT_MODEL,
) -> None:
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    # Apply styling
    sm = StyleManager(custom_styles)
    styled_prompt = sm.apply_style(prompt, style)

    # Validate image-to-image parameters
    if strength < 0.0 or strength > 1.0:
        raise ValueError(f"strength must be between 0.0 and 1.0, got {strength}")

    if init_image is not None:
        init_path = Path(init_image).expanduser().resolve()
        if not init_path.exists():
            raise FileNotFoundError(f"Reference image not found: {init_path}")
        if not init_path.is_file():
            raise ValueError(f"Reference image must be a file: {init_path}")

    # Resolve model-specific defaults
    defaults = MODEL_DEFAULTS.get(model_name.lower(), MODEL_DEFAULTS[DEFAULT_MODEL])
    steps = preset.get("steps", defaults["steps"])
    guidance = preset.get("guidance", defaults["guidance"])

    logger.debug(f"Using model '{model_name}' with {steps} steps, seed={seed}")

    # Use ModelManager for caching
    model = ModelManager.get_model(
        model_name=model_name,
        quantize=preset.get("quantize"),
    )

    # Build generate_image kwargs — common across all models
    gen_kwargs = dict(
        seed=seed,
        prompt=styled_prompt,
        num_inference_steps=steps,
        height=height,
        width=width,
        image_path=init_image,
        image_strength=strength,
    )

    # Add guidance only for models that support it
    if model_name.lower() != "zimage-turbo":
        gen_kwargs["guidance"] = guidance

    # Generate the image
    result = model.generate_image(**gen_kwargs)

    image = result.image if hasattr(result, "image") else result

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    logger.info(f"Image saved to {output_path}")
