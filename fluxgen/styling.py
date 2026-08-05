import logging
from typing import Dict, Optional, List

logger = logging.getLogger("fluxgen")

class StyleManager:
    """Manages prompt styling."""
    _default_styles = {
        "none": "",
        "ghibli": " in Studio Ghibli style, whimsical animation",
        "cinematic": " cinematic lighting, 8k resolution, highly detailed",
        "pixel": " in pixel art style, 16-bit aesthetic",
        "watercolor": " in watercolor painting style, soft washes, artistic",
        "anime": " in anime style, vibrant colors, detailed illustration",
        "photorealistic": " photorealistic, ultra detailed, DSLR quality, natural lighting",
        "oil-painting": " in oil painting style, rich textures, classical art",
        "comic": " in comic book style, bold lines, cel shading",
        "minimal": " minimalist design, clean lines, simple composition",
        "cyberpunk": " cyberpunk aesthetic, neon lights, futuristic, dark atmosphere",
    }

    def __init__(self, custom_styles: Optional[Dict[str, str]] = None):
        self.styles = StyleManager._default_styles if custom_styles is None else dict(StyleManager._default_styles, **custom_styles)

    def apply_style(self, prompt: str, style_name: str) -> str:
        if not style_name or style_name.lower() == "none":
            return prompt

        key = style_name.lower()
        suffix = self.styles.get(key)
        if suffix is None:
            # Unknown styles used to be silently swallowed (treated as
            # "none"), which turned typos like `--style gehibli` into
            # no-ops instead of surfacing the error. We keep the
            # fall-through behavior so a bad style does NOT mutate the
            # prompt, but emit a warning so the cause is visible.
            valid = ", ".join(sorted(self.styles))
            logger.warning(
                f"Unknown style '{style_name}'. Valid styles: {valid}. "
                f"Treating as no style."
            )
            return prompt

        return f"{prompt}{suffix}"

    def get_style_names(self) -> List[str]:
        return list(self.styles.keys())
