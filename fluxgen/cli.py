import argparse
import sys
from pathlib import Path
from fluxgen.generator import generate_image, generate_random_filename
from fluxgen.presets import PRESETS


def main():
    parser = argparse.ArgumentParser(description="Generate images using mflux")
    parser.add_argument("prompt", help="Text prompt for image generation")
    parser.add_argument(
        "-0",
        "--fast",
        action="store_const",
        const=0,
        dest="preset",
        help="Fast preset (default)",
    )
    parser.add_argument(
        "-3",
        "--standard",
        action="store_const",
        const=3,
        dest="preset",
        help="Standard preset",
    )
    parser.add_argument(
        "-8",
        "--quality",
        action="store_const",
        const=8,
        dest="preset",
        help="Quality preset",
    )
    parser.add_argument("--steps", type=int, help="Override steps")
    parser.add_argument("--quantize", type=int, help="Override quantize")
    parser.add_argument(
        "--output", help="Output file path (auto-generated if not specified)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for generated images (default: output)",
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--no-style", action="store_true", help="Disable default Ghibli style"
    )
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    # Image-to-image arguments
    parser.add_argument(
        "--init-image",
        "--image-path",
        type=str,
        dest="init_image",
        help="Path to reference image for image-to-image generation",
    )
    parser.add_argument(
        "--strength",
        "--image-strength",
        type=float,
        default=0.4,
        dest="strength",
        help="Strength of reference image influence (0.0-1.0, default: 0.4)",
    )

    args = parser.parse_args()

    # Generate random filename if output not specified
    output_path = args.output if args.output else generate_random_filename()
    output_path = str(Path(args.output_dir) / output_path)

    # Default preset
    preset_num = args.preset if args.preset is not None else 0
    preset = PRESETS[preset_num].copy()

    # Overrides
    if args.steps:
        preset["steps"] = args.steps
    if args.quantize:
        preset["quantize"] = args.quantize

    try:
        generate_image(
            prompt=args.prompt,
            preset=preset,
            seed=args.seed,
            output=output_path,
            width=args.width,
            height=args.height,
            no_style=args.no_style,
            init_image=args.init_image,
            strength=args.strength,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
