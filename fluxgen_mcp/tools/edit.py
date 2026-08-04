"""`edit_image` MCP tool.

Thin wrapper over `fluxgen.editor.ImageEditor.edit` that:
  - validates the prompt against the safety policy,
  - enforces input bounds (size, max dimension),
  - resolves the output path under the sandbox root,
  - delegates the actual edit to `asyncio.to_thread` so the MCP
    event loop stays responsive,
  - attaches seed/model/output_path metadata to `MCPError` so the
    server can build a complete audit record.

Mirrors `fluxgen_mcp.tools.generate` but for the edit pipeline.
Audit logging is owned by the server, so this module does not
write audit records.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from pathlib import Path
from typing import Any

from fluxgen.editor import ImageEditor
from fluxgen.generator import generate_random_filename

from fluxgen_mcp.config import MCPSettings
from fluxgen_mcp.errors import E_BAD_ARG, E_MODEL, MCPError, EXCEPTION_MAP
from fluxgen_mcp.safety import check_pause, resolve_sandbox_output, validate_prompt
from fluxgen_mcp.validation import validate_edit_inputs

logger = logging.getLogger("fluxgen-mcp")


_DEFAULT_EDIT_MODEL = "flux2-klein"


def _resolve_seed(seed: int | None) -> int:
    if seed is None:
        return random.randint(0, 2**32 - 1)
    return int(seed)


def _edit_filename(inputs: list[Path]) -> str:
    """Filename rule mirroring `fluxgen.cli.handle_edit`.

    Falls back to `generate_random_filename` (3 words + timestamp
    fallback for uniqueness) if `wonderwords` is missing. The
    wonderwords path uses a single short word which is weaker
    against collisions under sustained concurrent calls; the
    fallback is strictly safer. This matches the CLI's behavior
    verbatim; both should be migrated together if a stricter rule
    is adopted upstream.
    """
    base = inputs[0].stem or "edit"  # guard against empty stem (e.g. `/.png`)
    try:
        from wonderwords import RandomWord  # noqa: WPS433 — local import

        rw = RandomWord()
        suffix = rw.random_words(1, word_max_length=5)[0]
        return f"{base}_{suffix}.png"
    except ImportError:
        return generate_random_filename()


def _resolve_steps(steps: int | None, settings: MCPSettings) -> int | None:
    if steps is None:
        return None
    if steps < 1 or steps > settings.max_steps:
        raise MCPError(
            E_BAD_ARG,
            f"steps must be in [1, {settings.max_steps}]; got {steps}",
        )
    return int(steps)


def _resolve_guidance(guidance: float | None) -> float | None:
    if guidance is None:
        return None
    if guidance <= 0:
        raise MCPError(E_BAD_ARG, f"guidance must be > 0; got {guidance}")
    return float(guidance)


def _resolve_dim(d: int | None, settings: MCPSettings, axis: str) -> int | None:
    if d is None:
        return None
    cap = settings.max_width if axis == "width" else settings.max_height
    if d < 1 or d > cap:
        raise MCPError(
            E_BAD_ARG,
            f"{axis} must be in [1, {cap}]; got {d}",
        )
    return int(d)


async def edit_image_tool(
    *,
    settings: MCPSettings,
    input_paths: list[str],
    prompt: str,
    model: str | None,
    seed: int | None,
    steps: int | None,
    guidance: float | None,
    width: int | None,
    height: int | None,
    output_subdir: str,
) -> dict[str, Any]:
    """Run one edit. Returns a dict matching the tool's output schema.

    Raises:
        MCPError: any safety or input validation failure.
        fluxgen.exceptions.FluxgenError / OSError / RuntimeError:
            underlying pipeline errors are caught and re-raised as
            `MCPError(E_MODEL)`.
    """
    check_pause(settings)
    validate_prompt(settings, prompt)

    target_model = model or _DEFAULT_EDIT_MODEL
    if target_model not in settings.allowed_edit_models:
        raise MCPError(
            E_BAD_ARG,
            f"model {target_model!r} not in allowed_edit_models",
        )

    if not input_paths:
        raise MCPError(E_BAD_ARG, "input_paths must be non-empty")

    if target_model == "qwen-image-edit" and len(input_paths) > 1:
        # CLI rejects this at edit time too; replicate here so the
        # MCP layer fails fast without loading the pipeline.
        raise MCPError(
            E_BAD_ARG,
            "qwen-image-edit only supports a single input image",
        )

    resolved_inputs = validate_edit_inputs(
        input_paths,
        max_bytes=settings.input_max_bytes,
        max_dimension=settings.input_max_dimension,
    )

    final_steps = _resolve_steps(steps, settings)
    final_guidance = _resolve_guidance(guidance)
    final_w = _resolve_dim(width, settings, "width")
    final_h = _resolve_dim(height, settings, "height")
    final_seed = _resolve_seed(seed)

    output_dir = resolve_sandbox_output(settings, output_subdir)
    output_path = str(output_dir / _edit_filename(resolved_inputs))

    started = time.perf_counter()
    try:
        editor = ImageEditor(model_name=target_model)

        def _run_edit() -> None:
            # `_load_pipeline` is idempotent — both
            # `_load_qwen_pipeline` and `_load_mflux_pipeline` early-
            # return when their respective pipeline handle is already
            # populated (fluxgen/editor.py:60, fluxgen/editor.py:105).
            editor._load_pipeline()  # noqa: SLF001 — same pattern as CLI
            editor.edit(
                image_paths=[str(p) for p in resolved_inputs],
                prompt=prompt,
                output_path=output_path,
                steps=final_steps,
                guidance_scale=final_guidance,
                seed=final_seed,
                width=final_w,
                height=final_h,
            )

        await asyncio.to_thread(_run_edit)
    except tuple(EXCEPTION_MAP.keys()) as exc:
        code = EXCEPTION_MAP.get(type(exc), E_MODEL)
        raise MCPError(
            code,
            str(exc),
            seed=final_seed,
            model=target_model,
            output_path=None,
        ) from exc
    except (OSError, RuntimeError) as exc:
        # Model-load failures (missing GGUF / safetensors weights),
        # CUDA OOM, NaN outputs from the diffusion pipeline, etc.
        logger.warning(
            "tool edit_image: %s: %s", type(exc).__name__, exc,
        )
        raise MCPError(
            E_MODEL,
            f"{type(exc).__name__}: {exc}",
            seed=final_seed,
            model=target_model,
            output_path=None,
        ) from exc

    elapsed = time.perf_counter() - started
    return {
        "path": output_path,
        "elapsed_s": round(elapsed, 3),
        "model": target_model,
    }