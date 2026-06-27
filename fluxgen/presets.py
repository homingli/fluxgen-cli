from dataclasses import dataclass, asdict


@dataclass
class Preset:
    """A generation preset with inference parameters."""
    steps: int
    quantize: int
    guidance: float | None = None


PRESETS: dict[int, Preset] = {
    0: Preset(steps=5, quantize=8),
    3: Preset(steps=9, quantize=8),
    8: Preset(steps=16, quantize=16),
}


# Resolution presets: preset name -> (width, height)
# Smaller defaults = faster generation. 512x512 is the sweet spot for speed.
RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    "tiny":   (512, 512),
    "square": (768, 768),
    "large":  (1024, 1024),
    "full":   (1536, 1536),
}

# Aspect-ratio presets: preset name -> (width, height)
RESOLUTION_ASPECT_PRESETS: dict[str, tuple[int, int]] = {
    "1:1":    (512, 512),
    "16:9":   (960, 544),
    "9:16":   (544, 960),
    "4:3":    (768, 576),
    "3:4":    (576, 768),
}

# Merge all resolution presets (size + aspect) for a unified lookup
ALL_RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    **RESOLUTION_PRESETS,
    **RESOLUTION_ASPECT_PRESETS,
}
