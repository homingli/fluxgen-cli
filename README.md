# fluxgen-cli

A CLI tool for text-to-image and image-to-image generation using mflux, focused on speed and low GPU requirements.

## Installation

1. Ensure Python 3.10+ and pip are installed.
2. Install mflux: `pip install mflux`
3. Install this tool: `pip install -e /tmp/fluxgen-cli` (or from the directory)

## Usage

```bash
fluxgen --prompt "A beautiful landscape" -0  # Fast preset (default)
fluxgen --prompt "A city scene" -3 --steps 6  # Standard with override
fluxgen --prompt "A fantasy world" -8 --output fantasy.png --seed 123
fluxgen --prompt "A portrait" --output-dir portraits  # Save to portraits/ directory
fluxgen --prompt "Artistic" --output art/drawing.png  # Save to output/art/drawing.png
```

### Options
- `-0` (default): Fast preset
- `-3`: Standard preset
- `-8`: Quality preset
- `--steps INT`: Override steps
- `--quantize INT`: Override quantize
- `--output FILE`: Output file
- `--output-dir DIR`: Output directory for generated images (default: output)
- `--seed INT`: Random seed (auto if omitted)
- `--no-style`: Disable Ghibli style
- `--init-image` / `--image-path` FILE: Reference image for img2img
- `--strength` / `--image-strength` FLOAT: Influence strength (0.0-1.0, default 0.4)

## Presets
- 0: schnell, 2 steps, quantize 8
- 3: schnell, 4 steps, quantize 6
- 8: dev, 15 steps, quantize 8

## Changelog

### 0.1.1
- Added image-to-image generation support via `--init-image` / `--image-path`
- Added `--strength` / `--image-strength` parameter for reference image influence
- Added aliases matching mflux CLI naming conventions

### 0.1.0
- Added `--output-dir` option with default directory `output`
- Images now saved to output directory by default
- Output directory is created automatically if it doesn't exist