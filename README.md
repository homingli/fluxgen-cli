# fluxgen-cli

A CLI and Interactive REPL for local AI image generation and instruction-based image editing on macOS using mflux.

- **Generate images** using state-of-the-art `mflux` backends: `zimage-turbo`, `zimage`, `flux2-klein4b`, and `flux2-klein9b`.
- **Edit images** via natural language instructions using either `flux2-klein` (multi-image support!) or `qwen-image-edit` (GGUF weights).
- **Interactive REPL Mode** to keep models cached persistently in memory for near-instant successive runs.
- **Robust Path Security & Validation** protecting against directory traversal and corrupted image inputs.
- **Local Defaults** customizable in a simple `.fluxgen.toml` configuration.

---

## Installation

Install globally using `uv`:

```bash
uv tool install .
```

For local development and testing:

```bash
uv sync --dev
uv run fluxgen --help
```

---

## Requirements & Authentication

- Python 3.10+
- **Apple Silicon Mac** with 32 GB+ unified memory is highly recommended (especially for GGUF weights and MLX generation).
- **Hugging Face Hub Access** for model downloads.

### Hugging Face Authentication
The CLI downloads models (like `unsloth/Qwen-Image-Edit-2511-GGUF` or other weights) directly from the Hugging Face Hub. To authenticate:

1. Obtain a User Access Token from your [Hugging Face Security Tokens Settings](https://huggingface.co/docs/hub/en/security-tokens).
2. Log in using the Hugging Face CLI (automatically installed with dependencies):
   ```bash
   huggingface-cli login
   ```
3. Alternatively, export your token as an environment variable in your terminal session:
   ```bash
   export HF_TOKEN="your_huggingface_token"
   ```

> [!NOTE]
> Running the CLI will download models on first use. For example, the `qwen-image-edit` GGUF model is about **13 GB**, and the `flux2-klein` weights are also several gigabytes. Ensure you have a stable internet connection and sufficient disk space.

---

## Usage

If no subcommand is provided, `fluxgen` defaults to treating the first argument as a generation prompt.

### 1. Image Generation

```bash
fluxgen "A beautiful cinematic mountain landscape"
fluxgen gen "A bustling cyberpunk city scene" --preset standard
fluxgen gen "A fantasy world" --style cinematic
fluxgen gen "A playful puppy" --model zimage --timer
```

### 2. Instruction-Based Image Editing

The `edit` command supports two powerful editing models:

*   **`flux2-klein`** (default): Ultra-fast local editing optimized for MLX. **Supports editing multiple input images at once!**
    ```bash
    # Single image edit
    fluxgen edit photo.jpg "add a red hat to the cat"
    
    # Multi-image batch edit
    fluxgen edit shot1.png shot2.png shot3.png "make it sunset synthwave style" --model flux2-klein
    ```
*   **`qwen-image-edit`**: High-fidelity instruction editing utilizing GGUF weights via `diffusers`. Currently supports a **single input image**.
    ```bash
    fluxgen edit portrait.png "turn into an oil painting" --model qwen-image-edit --timer
    ```

### 3. Interactive REPL Mode

To completely eliminate the model loading overhead between consecutive generations or edits, start a persistent interactive shell:

```bash
fluxgen interactive
# Or:
fluxgen repl
```

Inside the REPL, models remain warmed up in memory:
```text
fluxgen> gen "A mystical forest" --style cinematic
fluxgen> edit output/generated-file.png "add a castle in the background" --model flux2-klein
fluxgen> help
```

---

## Command Reference

| Command | Usage | Description |
|---|---|---|
| `generate`, `gen` | `fluxgen gen "prompt"` | Generate an image from text or a reference image |
| `edit` | `fluxgen edit input.png "instruction"` | Edit existing image(s) using natural language instructions |
| `interactive`, `repl` | `fluxgen interactive` | Start an interactive REPL session (keeps models in memory) |

---

## CLI Options

### Global Options
- `-s`, `--silent`: Suppress non-error console output.
- `-v`, `--verbose`: Show debug output and full error tracebacks.

### Generation Options (`generate`, `gen`)
- `--model [zimage-turbo|zimage|flux2-klein4b|flux2-klein9b]`: Model backend to use (default: `zimage-turbo`).
- `-0`, `--fast`: Fast preset (fewer steps).
- `-3`, `--standard`: Standard preset.
- `-8`, `--quality`: Quality preset (higher steps).
- `--preset [fast|standard|quality]`: Named preset override.
- `--style NAME`: Built-in or configured prompt style (default: `none`).
- `--no-style`: Disable styling completely.
- `--steps INT`: Override preset inference steps.
- `--quantize INT`: Override mflux model quantization.
- `--width INT`, `--height INT`: Output dimensions (default: `1024x1024`).
- `--output FILE`: Output filename (auto-generated if not specified).
- `--output-dir DIR`: Output directory (default: `output`).
- `--seed INT`: Deterministic random seed.
- `--init-image FILE`: Reference image for image-to-image generation.
- `--strength FLOAT`: Reference image strength (default: `0.4`).
- `--timer`: Print elapsed generation time.

### Editing Options (`edit`)
- `--model [flux2-klein|qwen-image-edit]`: Editing model to use (default: `flux2-klein`).
- `--output FILE`: Output filename (default: first input image name appended with a random word).
- `--output-dir DIR`: Output directory (default: `output`).
- `--steps INT`: Inference steps override.
- `--guidance FLOAT`: Guidance scale override.
- `--quantize INT`: Override MLX quantization for `flux2-klein`.
- `--width INT`, `--height INT`: Force specific output dimensions (re-scales while preserving aspect ratio, max 1920px).
- `--seed INT`: Deterministic random seed.
- `--timer`: Print elapsed editing time.

---

## Configuration

Create a `.fluxgen.toml` configuration in your user home directory (`~/`) or the current project directory:

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

A local configuration in the current directory will automatically take precedence over the home directory configuration.

---

## Project Docs

- [Architecture](ARCHITECTURE.md)
- [Changelog](CHANGELOG.md)
