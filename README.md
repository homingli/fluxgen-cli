# fluxgen-cli

A CLI tool for text-to-image and image-to-image generation using mflux, focused on speed and low GPU requirements.

## Installation

1. Ensure Python 3.10+ and pip are installed.
2. Install mflux: `pip install mflux`
3. Install this tool: `pip install -e .` (from the directory)

## Usage

```bash
fluxgen --prompt "A beautiful landscape" -0             # Fast preset (default)
fluxgen --prompt "A city scene" --preset standard       # Named preset
fluxgen --prompt "A fantasy world" --style cinematic    # Cinematic style
fluxgen --prompt "Portrait" --style none                # Raw prompt (no style)
fluxgen --prompt "A portrait" --output-dir portraits    # Save to portraits/ directory
```

### Options
- `-0`, `--fast`: Fast preset (default)
- `-3`, `--standard`: Standard preset
- `-8`, `--quality`: Quality preset
- `--preset [fast|standard|quality]`: Named preset override
- `--style [ghibli|cinematic|none|...]`: Apply prompt styling
- `--no-style`: Shortcut for `--style none`
- `--steps INT`: Override steps
- `--quantize INT`: Override quantize
- `--output FILE`: Output file
- `--output-dir DIR`: Output directory (default: output)
- `--seed INT`: Random seed (auto if omitted)
- `--init-image` / `--image-path` FILE: Reference image for img2img
- `--strength` / `--image-strength` FLOAT: Influence strength (0.0-1.0, default 0.4)

## Configuration

You can create a `.fluxgen.toml` in the current directory or home directory to set persistent defaults and custom styles.

```toml
[defaults]
output_dir = "my_images"
width = 512
height = 512
preset = 0
style = "ghibli"

[styles]
ghibli = " in Studio Ghibli style, whimsical animation"
cinematic = " cinematic lighting, 8k resolution, highly detailed"
pixel = " in pixel art style, 16-bit aesthetic"
```

## Presets
- 0: schnell, 2 steps, quantize 8
- 3: schnell, 4 steps, quantize 6
- 8: dev, 15 steps, quantize 8

## Changelog

### 0.1.2
- Added support for `.fluxgen.toml` configuration file.
- Added named presets (`--preset fast`, `standard`, `quality`).
- Added flexible styling system via `--style` flag.
- Implemented model instance caching for programmatic use.
- Hardcoded Studio Ghibli styling moved to configurable style registry.

### 0.1.1
- Added image-to-image generation support via `--init-image` / `--image-path`
- Added `--strength` / `--image-strength` parameter for reference image influence
- Added aliases matching mflux CLI naming conventions

### 0.1.0
- Added `--output-dir` option with default directory `output`
- Images saved to output directory by default
- Output directory created automatically if it doesn't exist