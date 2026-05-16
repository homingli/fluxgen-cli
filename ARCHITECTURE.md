# Architecture

`fluxgen-cli` is a small Python CLI with two image workflows:

- generation through `mflux`
- instruction editing through Diffusers and Qwen-Image-Edit

## Entry Point

`fluxgen.cli:main` is the console script declared in `pyproject.toml`.

The CLI has two subcommands:

- `generate` / `gen`: text-to-image and image-to-image generation
- `edit`: instruction-based image editing

For backward compatibility, any first argument that is not a known subcommand is treated as a generation prompt.

## Generation Flow

Files:

- `fluxgen/cli.py`
- `fluxgen/generator.py`
- `fluxgen/presets.py`
- `fluxgen/styling.py`
- `fluxgen/config.py`

Flow:

1. `cli.py` parses generation arguments and config defaults.
2. `config.py` loads `.fluxgen.toml` from home and current directory.
3. `presets.py` supplies step and quantization presets.
4. `styling.py` applies built-in or configured prompt styles.
5. `generator.py` selects an `mflux` model through `ModelManager`.
6. The generated PIL image is saved to the requested output path.

`ModelManager` caches one active model instance. It recreates the model when the requested backend or quantization changes.

Supported generation backends:

- `zimage-turbo`
- `zimage`
- `flux1-schnell`

## Editing Flow

Files:

- `fluxgen/cli.py`
- `fluxgen/editor.py`

Flow:

1. `cli.py` builds the edit output path and creates `ImageEditor`.
2. `ImageEditor` chooses device priority: MPS, CUDA, then CPU.
3. The Qwen GGUF transformer is downloaded with `huggingface_hub`.
4. Diffusers loads `QwenImageEditPlusPipeline` from `Qwen/Qwen-Image-Edit-2511`.
5. The input image is converted to RGB and passed to the pipeline.
6. The resulting PIL image is validated and saved.

Editing uses:

- GGUF repo: `unsloth/Qwen-Image-Edit-2511-GGUF`
- GGUF file: `qwen-image-edit-2511-Q4_K_M.gguf`
- base config: `Qwen/Qwen-Image-Edit-2511`

The editor uses `bfloat16` on MPS and CUDA because Qwen-Image-Edit can produce invalid values with `float16`. CPU uses `float32`. Invalid Diffusers cast warnings or blank black images are treated as failures and are not saved.

## Configuration

Config filename: `.fluxgen.toml`

Load order:

1. home directory
2. current working directory

Current-directory values override home values. The config supports:

- `[defaults]` for CLI defaults
- `[styles]` for custom prompt style suffixes

## Output

Generation defaults to a random three-word PNG filename in the configured output directory.

Editing defaults to a random three-word PNG filename in the configured output directory.

Both flows create output directories as needed.

## Tests

Tests live in `tests/`.

Run:

```bash
.venv/bin/python -m pytest -q
```

The editor tests mock model loading and verify command wiring, dtype selection, defaults, and invalid-output rejection.
