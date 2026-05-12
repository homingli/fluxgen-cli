import argparse
import sys
import time
from pathlib import Path
try:
    from importlib.metadata import distribution
except ImportError:
    from importlib_metadata import distribution  # Python < 3.8 fallback

from fluxgen.generator import generate_image, generate_random_filename, SUPPORTED_MODELS, DEFAULT_MODEL
from fluxgen.presets import PRESETS
from fluxgen.config import load_config, get_config_value

def main():
    config = load_config()
    
    # Get version from pyproject.toml
    try:
        dist = distribution("fluxgen-cli")
        version = dist.version
    except Exception:
        version = "0.2.0"
    
    parser = argparse.ArgumentParser(description=f"fluxgen v{version} - AI Image Generation & Editing")
    parser.add_argument("--version", action="version", version=f"fluxgen {version}")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── GENERATE COMMAND ──────────────────────────────────────────────────────
    gen_parser = subparsers.add_parser("generate", aliases=["gen"], help="Generate an image from text")
    gen_parser.add_argument("prompt", help="Text prompt for image generation")
    
    # Presets
    gen_parser.add_argument(
        "-0", "--fast", action="store_const", const=0, dest="preset_idx",
        help="Fast preset (default)"
    )
    gen_parser.add_argument(
        "-3", "--standard", action="store_const", const=3, dest="preset_idx",
        help="Standard preset"
    )
    gen_parser.add_argument(
        "-8", "--quality", action="store_const", const=8, dest="preset_idx",
        help="Quality preset"
    )
    gen_parser.add_argument(
        "--preset", choices=["fast", "standard", "quality"], 
        help="Named preset (overrides numeric flags)"
    )

    gen_parser.add_argument("--steps", type=int, help="Override steps")
    gen_parser.add_argument("--quantize", type=int, help="Override quantize")
    gen_parser.add_argument(
        "--output", help="Output file path (auto-generated if not specified)"
    )
    gen_parser.add_argument(
        "--output-dir", type=str, 
        default=get_config_value(config, "output_dir", "output"),
        help="Output directory (default: output)"
    )
    gen_parser.add_argument("--seed", type=int, help="Random seed")
    gen_parser.add_argument(
        "--style", type=str, 
        default=get_config_value(config, "style", "none"),
        help="Style to apply (default: none)"
    )
    gen_parser.add_argument(
        "--no-style", action="store_const", const="none", dest="style",
        help="Disable styling"
    )
    gen_parser.add_argument(
        "--model", type=str, choices=SUPPORTED_MODELS,
        default=get_config_value(config, "model", DEFAULT_MODEL),
        help=f"Model to use (default: {DEFAULT_MODEL})"
    )
    gen_parser.add_argument("--width", type=int, default=get_config_value(config, "width", 1024))
    gen_parser.add_argument("--height", type=int, default=get_config_value(config, "height", 1024))
    gen_parser.add_argument("--init-image", type=str, help="Reference image for img2img")
    gen_parser.add_argument("--strength", type=float, default=0.4, help="Img2img strength")
    gen_parser.add_argument("--timer", action="store_true", help="Show generation time")

    # ── EDIT COMMAND ──────────────────────────────────────────────────────────
    edit_parser = subparsers.add_parser("edit", help="Edit an image using instructions (Qwen-Image-Edit)")
    edit_parser.add_argument("image", help="Path to the input image")
    edit_parser.add_argument("prompt", help="Instruction for the edit (e.g., 'add a red hat')")
    edit_parser.add_argument("--output", help="Output filename (saved in output dir)")
    edit_parser.add_argument(
        "--output-dir", type=str,
        default=get_config_value(config, "output_dir", "output"),
        help="Output directory (default: output)"
    )
    edit_parser.add_argument("--steps", type=int, default=None, help="Override inference steps (default: 40)")
    edit_parser.add_argument("--guidance", type=float, default=1.0, help="Guidance scale (default: 1.0)")
    edit_parser.add_argument("--timer", action="store_true", help="Show execution time")

    # ── BACKWARD COMPATIBILITY ────────────────────────────────────────────────
    # If no recognized command is given and there's at least one argument, default to 'generate'
    if len(sys.argv) > 1 and sys.argv[1] not in ["generate", "gen", "edit", "--version", "--help", "-h"]:
        sys.argv.insert(1, "generate")

    args = parser.parse_args()

    if args.command in ["generate", "gen"]:
        handle_generate(args, config)
    elif args.command == "edit":
        handle_edit(args)
    else:
        parser.print_help()

def handle_generate(args, config):
    # Determine preset index
    named_indices = {"fast": 0, "standard": 3, "quality": 8}
    preset_idx = args.preset_idx
    if args.preset:
        preset_idx = named_indices[args.preset]
    
    if preset_idx is None:
        preset_idx = get_config_value(config, "preset", 0)
    
    preset = PRESETS[preset_idx].copy()
    if args.steps:
        preset["steps"] = args.steps
    if args.quantize:
        preset["quantize"] = args.quantize

    output_path = args.output if args.output else generate_random_filename()
    output_path = str(Path(args.output_dir) / output_path)

    try:
        start = time.perf_counter() if args.timer else None
        generate_image(
            prompt=args.prompt,
            preset=preset,
            seed=args.seed,
            output=output_path,
            width=args.width,
            height=args.height,
            style=args.style,
            custom_styles=config.get("styles"),
            init_image=args.init_image,
            strength=args.strength,
            model_name=args.model,
        )
        if start is not None:
            elapsed = time.perf_counter() - start
            print(f"⏱ Generated in {elapsed:.2f}s")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def handle_edit(args):
    try:
        from fluxgen.editor import ImageEditor, EDIT_DEFAULT_STEPS

        # Build output path inside the output directory
        if args.output:
            output_filename = args.output
        else:
            input_path = Path(args.image)
            output_filename = f"edited_{input_path.name}"
        output_path = str(Path(args.output_dir) / output_filename)

        # Use the editor's default if --steps is not provided
        steps = args.steps if args.steps is not None else EDIT_DEFAULT_STEPS
        
        start = time.perf_counter() if args.timer else None
        
        print("\nNOTE: First run will download the Q4_K_M GGUF model (~13GB).")
        print("This requires disk space and a stable connection.\n")
        
        editor = ImageEditor()
        editor.edit(
            image_path=args.image,
            prompt=args.prompt,
            output_path=output_path,
            steps=steps,
            guidance_scale=args.guidance,
        )
        
        if start is not None:
            elapsed = time.perf_counter() - start
            print(f"⏱ Edited in {elapsed:.2f}s")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
