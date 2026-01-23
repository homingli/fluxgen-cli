import random
import time
from pathlib import Path
from mflux.models.flux.variants.txt2img.flux import Flux1


def generate_random_filename() -> str:
    """Generate a random 3-word filename with .png extension"""
    try:
        from wonderwords import RandomWord

        rw = RandomWord()
        words = rw.random_words(3, word_max_length=5)
        return "-".join(words) + ".png"
    except Exception:
        # Fallback to timestamp-based name if wonderwords fails
        import time

        return f"generated-{int(time.time())}.png"


def generate_image(
    prompt: str,
    preset: dict,
    seed: int | None = None,
    output: str = "output.png",
    width: int = 512,
    height: int = 512,
    no_style: bool = False,
    init_image: str | None = None,
    strength: float = 0.4,
) -> None:
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    if not no_style:
        prompt += " in Studio Ghibli style, whimsical animation"

    # Validate image-to-image parameters
    if strength < 0.0 or strength > 1.0:
        raise ValueError(f"strength must be between 0.0 and 1.0, got {strength}")

    if init_image is not None:
        init_path = Path(init_image)
        if not init_path.exists():
            raise FileNotFoundError(f"Reference image not found: {init_image}")
        if not init_path.is_file():
            raise ValueError(f"Reference image must be a file: {init_image}")

    flux = Flux1.from_name(
        model_name=preset["model"],
        quantize=preset["quantize"],
    )

    image = flux.generate_image(
        seed=seed,
        prompt=prompt,
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
