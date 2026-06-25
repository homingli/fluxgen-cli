from typing import Dict, Optional, List

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
            
        suffix = self.styles.get(style_name.lower())
        if suffix is None:
            # If style not found, treat it as "none"
            return prompt
            
        return f"{prompt}{suffix}"

    def get_style_names(self) -> List[str]:
        return list(self.styles.keys())
