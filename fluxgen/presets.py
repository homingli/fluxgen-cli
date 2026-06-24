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
