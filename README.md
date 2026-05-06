# fluxgen-cli

A CLI tool for text-to-image and image-to-image generation using [mflux](https://github.com/filipstrand/mflux), focused on speed and low GPU requirements. Supports multiple model backends including ZImage Turbo, ZImage, and FLUX.1 Schnell.

## Installation

1. Install [uv](https://github.com/astral-sh/uv) if not already installed.
2. Install the CLI: `uv tool install .`

### Requirements

- **HuggingFace Token**: You must have a HuggingFace token to download the models. Set it as environment variable:
  ```bash
  export HF_TOKEN="your_huggingface_token"
  ```
  Or log in with `hf login` after installing `huggingface-hub`.

## Usage

```bash
fluxgen "A beautiful landscape" -0                      # Fast preset (default, ZImage Turbo)
fluxgen "A city scene" --preset standard                # Named preset
fluxgen "A fantasy world" --style cinematic             # Cinematic style
fluxgen "Portrait" --style none                         # Raw prompt (no style)
fluxgen "A portrait" --output-dir portraits             # Save to portraits/ directory
fluxgen "A cat" --model zimage                          # Use ZImage (guidance-enabled)
fluxgen "A sunset" --model flux1-schnell                # Use FLUX.1 Schnell
fluxgen "A dog" --timer                                 # Show generation time
```

### Options
- `--model [zimage-turbo|zimage|flux1-schnell]`: Model backend (default: zimage-turbo)
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
- `--width INT`: Image width
- `--height INT`: Image height
- `--init-image` / `--image-path` FILE: Reference image for img2img
- `--strength` / `--image-strength` FLOAT: Influence strength (0.0-1.0, default 0.4)
- `--timer`: Show how long image generation took (off by default)

## Models

| Model | ID | Guidance | Default Steps | Description |
|---|---|---|---|---|
| ZImage Turbo | `zimage-turbo` | ✗ | 4 | Fast, guidance-free generation (default) |
| ZImage | `zimage` | ✓ | 20 | Full quality with classifier-free guidance |
| FLUX.1 Schnell | `flux1-schnell` | ✗ | 4 | Black Forest Labs distilled model |

## Configuration

You can create a `.fluxgen.toml` in the current directory or home directory to set persistent defaults and custom styles.

```toml
[defaults]
model = "zimage-turbo"
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
- 0 (`fast`): 5 steps, quantize 8, guidance 4.0
- 3 (`standard`): 9 steps, quantize 6, guidance 4.0
- 8 (`quality`): 15 steps, quantize 8, guidance 4.0

> **Note:** Guidance values from presets are only applied to models that support it (e.g. `zimage`). Guidance-free models like `zimage-turbo` and `flux1-schnell` ignore the guidance parameter.

## Changelog

### 0.1.4
- Added `--timer` flag to measure and display image generation time.
- Timer is off by default; when enabled, prints elapsed time (e.g. `⏱ Generated in 12.34s`).

### 0.1.3
- Added multi-model support via `--model` flag.
- Supported models: ZImage Turbo (default), ZImage, and FLUX.1 Schnell.
- Model-specific default parameters (guidance, steps) applied automatically.
- Model selection configurable via `.fluxgen.toml` (`model` key in `[defaults]`).
- ModelManager refactored with lazy imports and automatic caching per model+quantize config.

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
