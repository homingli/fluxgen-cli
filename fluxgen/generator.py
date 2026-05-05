import random
from pathlib import Path
from PIL import Image

from mflux.models.common.config import ModelConfig

# Supported model identifiers
SUPPORTED_MODELS = ["zimage-turbo", "zimage", "flux1-schnell"]
DEFAULT_MODEL = "zimage-turbo"


class ModelManager:
    """Manages model instances with caching and multi-model support.

    Supported models:
      - zimage-turbo  (default) — fast, guidance-free ZImage variant
      - zimage                  — full ZImage with guidance support
      - flux1-schnell           — FLUX.1 Schnell text-to-image
    """

    _instance = None
    _current_config = None

    @classmethod
    def get_model(cls, model_name: str, quantize: int | None = None):
        """Return a cached model instance, re-creating only when config changes."""
        model_name = model_name.lower()
        if model_name not in SUPPORTED_MODELS:
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
        if model_name == "flux1-schnell":
            from mflux.models.flux.variants.txt2img.flux import Flux1

            return Flux1(
                quantize=quantize,
                model_config=ModelConfig.schnell(),
            )
        elif model_name == "zimage":
            from mflux.models.z_image import ZImage

            return ZImage(
                quantize=quantize,
                model_config=ModelConfig.z_image(),
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
    "flux1-schnell": {
        "guidance": 0.0,       # schnell ignores guidance
        "steps": 4,
    },
}


class StyleManager:
    """Manages prompt styling."""
    DEFAULT_STYLES = {
        "ghibli": " in Studio Ghibli style, whimsical animation",
        "cinematic": " cinematic lighting, 8k resolution, highly detailed",
        "none": ""
    }

    def __init__(self, custom_styles: dict[str, str] | None = None):
        self.styles = self.DEFAULT_STYLES.copy()
        if custom_styles:
            self.styles.update(custom_styles)

    def apply_style(self, prompt: str, style_name: str) -> str:
        suffix = self.styles.get(style_name.lower())
        if suffix is None:
            # If style not found, treat it as "none" or maybe just return as is
            return prompt
        return f"{prompt}{suffix}"

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
        init_path = Path(init_image)
        if not init_path.exists():
            raise FileNotFoundError(f"Reference image not found: {init_image}")
        if not init_path.is_file():
            raise ValueError(f"Reference image must be a file: {init_image}")

    # Resolve model-specific defaults
    defaults = MODEL_DEFAULTS.get(model_name.lower(), MODEL_DEFAULTS[DEFAULT_MODEL])
    steps = preset.get("steps", defaults["steps"])
    guidance = preset.get("guidance", defaults["guidance"])

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

    result = model.generate_image(**gen_kwargs)

    # Flux1 returns a GeneratedImage wrapper; extract the PIL image
    if hasattr(result, "image"):
        image = result.image
    else:
        image = result

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Image saved to {output_path}")
