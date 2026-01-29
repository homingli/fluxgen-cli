import argparse
import sys
from pathlib import Path
from fluxgen.generator import generate_image, generate_random_filename
from fluxgen.presets import PRESETS
from fluxgen.config import load_config, get_config_value

def main():
    config = load_config()
    
    parser = argparse.ArgumentParser(description="Generate images using mflux")
    parser.add_argument("prompt", help="Text prompt for image generation")
    
    # Presets
    parser.add_argument(
        "-0", "--fast", action="store_const", const=0, dest="preset_idx",
        help="Fast preset (default)"
    )
    parser.add_argument(
        "-3", "--standard", action="store_const", const=3, dest="preset_idx",
        help="Standard preset"
    )
    parser.add_argument(
        "-8", "--quality", action="store_const", const=8, dest="preset_idx",
        help="Quality preset"
    )
    parser.add_argument(
        "--preset", choices=["fast", "standard", "quality"], 
        help="Named preset (overrides numeric flags)"
    )

    parser.add_argument("--steps", type=int, help="Override steps")
    parser.add_argument("--quantize", type=int, help="Override quantize")
    parser.add_argument(
        "--output", help="Output file path (auto-generated if not specified)"
    )
    parser.add_argument(
        "--output-dir", type=str, 
        default=get_config_value(config, "output_dir", "output"),
        help="Output directory for generated images (default: output)"
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--style", type=str, 
        default=get_config_value(config, "style", "ghibli"),
        help="Style to apply (default: ghibli, use 'none' to disable)"
    )
    parser.add_argument(
        "--no-style", action="store_const", const="none", dest="style",
        help="Disable styling (alias for --style none)"
    )
    parser.add_argument(
        "--width", type=int, 
        default=get_config_value(config, "width", 1024), 
        help="Image width"
    )
    parser.add_argument(
        "--height", type=int, 
        default=get_config_value(config, "height", 1024), 
        help="Image height"
    )

    # Image-to-image arguments
    parser.add_argument(
        "--init-image", "--image-path", type=str, dest="init_image",
        help="Path to reference image for image-to-image generation"
    )
    parser.add_argument(
        "--strength", "--image-strength", type=float, default=0.4, dest="strength",
        help="Strength of reference image influence (0.0-1.0, default: 0.4)"
    )

    args = parser.parse_args()

    # Determine preset index
    named_indices = {"fast": 0, "standard": 3, "quality": 8}
    preset_idx = args.preset_idx
    if args.preset:
        preset_idx = named_indices[args.preset]
    
    # Fallback to config default or 0
    if preset_idx is None:
        preset_idx = get_config_value(config, "preset", 0)
    
    preset = PRESETS[preset_idx].copy()

    # Overrides
    if args.steps:
        preset["steps"] = args.steps
    if args.quantize:
        preset["quantize"] = args.quantize

    # Output path logic
    output_path = args.output if args.output else generate_random_filename()
    output_path = str(Path(args.output_dir) / output_path)

    try:
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
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
