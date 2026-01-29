import random
from pathlib import Path
from mflux.models.flux.variants.txt2img.flux import Flux1

class ModelManager:
    """Manages Flux1 model instances to allow caching in programmatic use."""
    _instance = None
    _current_config = None

    @classmethod
    def get_model(cls, model_name: str, quantize: int) -> Flux1:
        config = (model_name, quantize)
        if cls._instance is None or cls._current_config != config:
            cls._instance = Flux1.from_name(
                model_name=model_name,
                quantize=quantize,
            )
            cls._current_config = config
        return cls._instance

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

    # Use ModelManager for caching
    flux = ModelManager.get_model(
        model_name=preset["model"],
        quantize=preset["quantize"],
    )

    image = flux.generate_image(
        seed=seed,
        prompt=styled_prompt,
        num_inference_steps=preset["steps"],
        guidance=preset["guidance"],
        height=height,
        width=width,
        image_path=init_image,
        image_strength=strength,
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Image saved to {output_path}")
