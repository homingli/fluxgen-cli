# Changelog

## 0.3.0 - 2026-05-16

- Removed `flux1-schnell` model from supported backends (deprecated).
- Split `flux2-klein` into `flux2-klein4b` (4B, default) and `flux2-klein9b` (9B). `flux2-klein` identifier removed.

## 0.2.0 - 2026-05-12

- Changed edit default output names to random Wonderwords filenames.
- Added Qwen-Image-Edit 2511 editing through Diffusers with GGUF transformer weights.
- Added `fluxgen edit image prompt` command for instruction-based image editing.
- Added accelerator-aware edit dtype handling: `bfloat16` on MPS/CUDA, `float32` on CPU.
- Added invalid edit output detection so NaN/blank black results fail instead of being saved.
- Added editor tests for pipeline wiring, defaults, dtype selection, and blank-output rejection.
- Refreshed README and added architecture documentation.
- Added `flux2-klein` (FLUX.2 Klein 9B) to supported models.

## 0.1.6

- Added initial `edit` command using Qwen-Image-Edit for instruction-based image editing.
- Refactored CLI to support subcommands with backward-compatible generation prompts.
- Added `fluxgen/editor.py` with MPS and CUDA device selection.
- Added dependencies: `torch`, `diffusers`, `transformers`, `accelerate`, `torchvision`, and `gguf`.

## 0.1.5

- Changed `--style` default to `none`.
- Added 8 built-in styles.
- Refactored `StyleManager`.

## 0.1.4

- Added `--timer` flag for generation timing.

## 0.1.3

- Added multi-model support through `--model`: `zimage-turbo`, `zimage`, and `flux1-schnell`.
