# fluxgen-cli

CLI for local AI image generation and instruction-based image editing.

- Generate images with `mflux` backends: `zimage-turbo`, `zimage`, `flux2-klein4b`, and `flux2-klein9b`.
- Edit existing images with Qwen-Image-Edit 2511 through Diffusers and GGUF weights.
- Keep local defaults in `.fluxgen.toml`.

## Installation

Install with `uv`:

```bash
uv tool install .
```

For development:

```bash
uv sync --dev
uv run fluxgen --help
```

## Requirements

- Python 3.10+
- Hugging Face access for model downloads:

```bash
export HF_TOKEN="your_huggingface_token"
```

Editing downloads `unsloth/Qwen-Image-Edit-2511-GGUF` on first use. The Q4_K_M GGUF file is about 13 GB, and the full edit stack still needs substantial RAM or unified memory. Apple Silicon with 32 GB+ unified memory is recommended.

## Usage

Generate:

```bash
fluxgen "A beautiful landscape"
fluxgen gen "A city scene" --preset standard
fluxgen gen "A fantasy world" --style cinematic
fluxgen gen "A cat" --model zimage
fluxgen gen "A dog" --timer
```

Image-to-image:

```bash
fluxgen gen "turn this into a watercolor poster" --init-image input.png --strength 0.45
```

Edit:

```bash
fluxgen edit input.png "make it sunset"
fluxgen edit photo.jpg "add a red hat to the cat"
fluxgen edit portrait.png "turn into an oil painting"
fluxgen edit scene.png "remove the tree" --timer
```

If no command is provided, `fluxgen` treats the first argument as a generation prompt.

## Commands

| Command | Usage | Description |
|---|---|---|
| `generate`, `gen` | `fluxgen gen "prompt"` | Generate an image from text or a reference image |
| `edit` | `fluxgen edit image.png "instruction"` | Edit an existing image with Qwen-Image-Edit |

## Global Options

- `-s`, `--silent`: suppress non-error output
- `-v`, `--verbose`: show debug output

## Generation Options

- `--model [zimage-turbo|zimage|flux2-klein4b|flux2-klein9b]`: model backend, default `zimage-turbo`
- `-0`, `--fast`: fast preset, default
- `-3`, `--standard`: standard preset
- `-8`, `--quality`: quality preset
- `--preset [fast|standard|quality]`: named preset override
- `--style NAME`: built-in or configured style, default `none`
- `--no-style`: disable prompt styling
- `--steps INT`: override inference steps
- `--quantize INT`: override mflux quantization
- `--width INT`, `--height INT`: image size, default `1024x1024`
- `--output FILE`: output filename
- `--output-dir DIR`: output directory, default `output`
- `--seed INT`: deterministic seed
- `--init-image FILE`: reference image for image-to-image
- `--strength FLOAT`: image-to-image strength, default `0.4`
- `--timer`: print elapsed time

## Editing Options

- `--output FILE`: output filename, default random three-word PNG
- `--output-dir DIR`: output directory, default `output`
- `--steps INT`: inference steps, default `40`
- `--guidance FLOAT`: guidance scale, default `1.0`
- `--timer`: print elapsed time

The editor uses `bfloat16` on CUDA and MPS, and `float32` on CPU. If Diffusers produces invalid values or a blank black output, the CLI raises an error instead of saving a bad image.

## Configuration

Create `.fluxgen.toml` in the current directory or home directory.

```toml
[defaults]
model = "zimage-turbo"
output_dir = "my_images"
width = 1024
height = 1024
preset = 0
style = "none"

[styles]
retro = " in retro 80s style, synthwave colors"
```

Local config overrides home config for matching keys.

## Project Docs

- [Architecture](ARCHITECTURE.md)
- [Changelog](CHANGELOG.md)
