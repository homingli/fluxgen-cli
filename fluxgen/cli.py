import argparse
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
import logging
import os
import sys
import time
from pathlib import Path
try:
    from importlib.metadata import distribution
except ImportError:
    from importlib_metadata import distribution  # Python < 3.8 fallback

from fluxgen.generator import generate_image, generate_random_filename, SUPPORTED_MODELS, DEFAULT_MODEL
from fluxgen.presets import PRESETS, ALL_RESOLUTION_PRESETS
from dataclasses import asdict
from fluxgen.config import load_config, get_config_value

logger = logging.getLogger("fluxgen")


GLOBAL_FLAGS = {"-v", "--verbose", "-s", "--silent"}
COMMANDS = {"generate", "gen", "edit", "interactive", "repl"}
PASSTHROUGH_FLAGS = {"--version", "--help", "-h"}

class ParserExit(Exception):
    pass

class InteractiveParser(argparse.ArgumentParser):
    def exit(self, status=0, message=None):
        if message:
            self._print_message(message, sys.stderr)
        raise ParserExit()

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n")


def setup_logging(verbose=False, silent=False):
    level = logging.DEBUG if verbose else logging.ERROR if silent else logging.INFO
    handler = logging.StreamHandler()
    handler.setLevel(level)
    fmt = "%(levelname)s: %(message)s" if verbose else "%(message)s"
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def add_verbosity_flags(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Show debug output",
    )
    group.add_argument(
        "-s",
        "--silent",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Suppress non-error output",
    )


def with_default_command(argv):
    insert_at = 0
    while insert_at < len(argv) and argv[insert_at] in GLOBAL_FLAGS:
        insert_at += 1

    if insert_at == len(argv):
        return argv

    token = argv[insert_at]
    if token in COMMANDS or token in PASSTHROUGH_FLAGS:
        return argv

    return argv[:insert_at] + ["generate"] + argv[insert_at:]


@contextmanager
def suppress_external_output(enabled):
    if not enabled:
        with nullcontext():
            yield
        return

    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def get_parser(config, version, interactive=False):
    verbosity_parent = argparse.ArgumentParser(add_help=False)
    add_verbosity_flags(verbosity_parent)

    parser_cls = InteractiveParser if interactive else argparse.ArgumentParser
    parser = parser_cls(
        description=f"fluxgen v{version} - AI Image Generation & Editing",
        parents=[verbosity_parent],
    )
    if not interactive:
        parser.add_argument("--version", action="version", version=f"fluxgen {version}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── GENERATE COMMAND ──────────────────────────────────────────────────────
    gen_parser = subparsers.add_parser(
        "generate",
        aliases=["gen"],
        help="Generate an image from text",
        parents=[verbosity_parent],
    )
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
    gen_parser.add_argument(
        "--resolution", "-r",
        type=str,
        choices=list(ALL_RESOLUTION_PRESETS.keys()),
        default=argparse.SUPPRESS,
        help="Resolution preset (default: tiny 512x512 for faster generation)",
    )
    gen_parser.add_argument("--width", type=int, default=argparse.SUPPRESS, help="Image width (overrides --resolution)")
    gen_parser.add_argument("--height", type=int, default=argparse.SUPPRESS, help="Image height (overrides --resolution)")
    gen_parser.add_argument("--init-image", type=str, help="Reference image for img2img")
    gen_parser.add_argument("--strength", type=float, default=0.4, help="Img2img strength")
    gen_parser.add_argument("--timer", action="store_true", help="Show generation time")

    # ── EDIT COMMAND ──────────────────────────────────────────────────────────
    edit_parser = subparsers.add_parser(
        "edit",
        help="Edit an image using instructions (Qwen-Image-Edit)",
        parents=[verbosity_parent],
    )
    edit_parser.add_argument("image", nargs="+", help="Path to the input image(s)")
    edit_parser.add_argument("prompt", help="Instruction for the edit (e.g., 'add a red hat')")
    edit_parser.add_argument("--output", help="Output filename (saved in output dir)")
    edit_parser.add_argument(
        "--output-dir", type=str,
        default=get_config_value(config, "output_dir", "output"),
        help="Output directory (default: output)"
    )
    edit_parser.add_argument(
        "--model", type=str, choices=["qwen-image-edit", "flux2-klein"],
        default="flux2-klein",
        help="Model to use for editing (default: flux2-klein)"
    )
    edit_parser.add_argument("--quantize", type=int, help="Override quantize for flux2-klein")
    edit_parser.add_argument("--seed", type=int, help="Random seed")
    edit_parser.add_argument("--steps", type=int, default=None, help="Override inference steps")
    edit_parser.add_argument("--guidance", type=float, default=None, help="Guidance scale")
    edit_parser.add_argument("--width", type=int, help="Output image width (defaults to input image width)")
    edit_parser.add_argument("--height", "--length", type=int, dest="height", help="Output image height/length (defaults to input image height/length)")
    edit_parser.add_argument("--timer", action="store_true", help="Show execution time")

    # ── INTERACTIVE COMMAND ───────────────────────────────────────────────────
    subparsers.add_parser(
        "interactive",
        aliases=["repl"],
        help="Start an interactive session to keep models loaded in memory",
        parents=[verbosity_parent],
    )

    return parser


def main(argv=None):
    config = load_config()

    # Get version from pyproject.toml
    try:
        dist = distribution("fluxgen-cli")
        version = dist.version
    except (ImportError, FileNotFoundError):
        version = "0.3.1"

    parser = get_parser(config, version)

    args = parser.parse_args(with_default_command(list(sys.argv[1:] if argv is None else argv)))

    if getattr(args, "verbose", False) and getattr(args, "silent", False):
        parser.error("argument -s/--silent: not allowed with argument -v/--verbose")

    # Resolve verbosity/silent and apply logging
    silent = getattr(args, "silent", False)
    setup_logging(verbose=getattr(args, "verbose", False), silent=silent)

    with suppress_external_output(silent):
        if args.command in ["generate", "gen"]:
            handle_generate(args, config)
        elif args.command == "edit":
            handle_edit(args)
        elif args.command in ["interactive", "repl"]:
            handle_interactive(config, version)
        else:
            parser.print_help()

def handle_interactive(config, version):
    import shlex
    try:
        import readline
    except ImportError:
        pass

    print(r"""
  __ _                               
 / _| |                              
| |_| |_   ___  ____ _  ___ _ __     
|  _| | | | \ \/ / _` |/ _ \ '_ \    
| | | | |_| |>  < (_| |  __/ | | |   
|_| |_|\__,_/_/\_\__, |\___|_| |_|   
                  __/ |              
                 |___/               
""")
    logger.info("Starting fluxgen interactive mode. Type 'exit', 'quit' or 'help' to navigate.")
    parser = get_parser(config, version, interactive=True)
    
    while True:
        try:
            cmd = input("\nfluxgen> ").strip()
            if not cmd:
                continue
            if cmd.lower() in ["exit", "quit"]:
                break
                
            argv = shlex.split(cmd)
            if not argv:
                continue
                
            if argv[0] in ["help", "-h", "--help"]:
                parser.print_help()
                continue
                
            if argv[0] not in COMMANDS:
                continue
                
            try:
                args = parser.parse_args(argv)
            except ParserExit:
                continue
                
            silent = getattr(args, "silent", False)
            setup_logging(verbose=getattr(args, "verbose", False), silent=silent)
            
            with suppress_external_output(silent):
                if args.command in ["generate", "gen"]:
                    handle_generate(args, config, interactive=True)
                elif args.command == "edit":
                    handle_edit(args, interactive=True)
                elif args.command in ["interactive", "repl"]:
                    logger.error("Already in interactive mode.")
                else:
                    parser.print_help()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        except Exception as e:
            logger.error(f"Error: {e}")

from contextlib import contextmanager
from fluxgen.exceptions import FluxgenError, PathTraversalError

@contextmanager
def error_handler(args, interactive=False):
    try:
        yield
    except (FileNotFoundError, ValueError, FluxgenError) as e:
        logger.error(f"Error: {e}")
        if getattr(args, "verbose", False):
            import traceback
            traceback.print_exc()
        if not interactive:
            sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if getattr(args, "verbose", False):
            import traceback
            traceback.print_exc()
        if not interactive:
            sys.exit(1)

def resolve_output_path(output_arg, output_dir_arg, default_filename_func=None):
    output_dir = Path(output_dir_arg).expanduser().resolve()
    
    if output_arg:
        output_path = Path(output_arg)
        if output_path.is_absolute():
            final_path = output_path.resolve()
        else:
            final_path = (output_dir / output_path).resolve()
            if not final_path.is_relative_to(output_dir):
                raise PathTraversalError(f"Output path {final_path} is outside the allowed directory {output_dir}")
    else:
        if default_filename_func:
            output_filename = default_filename_func()
        else:
            from fluxgen.generator import generate_random_filename
            output_filename = generate_random_filename()
        final_path = (output_dir / output_filename).resolve()
        
    return str(final_path)

def handle_generate(args, config, interactive=False):
    with error_handler(args, interactive):
        # Determine preset index
        named_indices = {"fast": 0, "standard": 3, "quality": 8}
        preset_idx = args.preset_idx
        if args.preset:
            preset_idx = named_indices[args.preset]

        if preset_idx is None:
            preset_idx = get_config_value(config, "preset", 0)

        preset = asdict(PRESETS[preset_idx])
        if args.steps:
            preset["steps"] = args.steps
        if args.quantize:
            preset["quantize"] = args.quantize

        output_path = resolve_output_path(args.output, args.output_dir)

        # Resolve resolution preset -> actual dimensions
        # Priority: explicit CLI --width/--height > explicit --resolution > config > default
        cli_resolution = getattr(args, "resolution", None)
        config_width = get_config_value(config, "width", None)
        config_height = get_config_value(config, "height", None)
        cli_width = getattr(args, "width", None)
        cli_height = getattr(args, "height", None)

        if cli_width is not None or cli_height is not None:
            # Explicit --width/--height wins; missing axis falls through chain
            if cli_width is not None:
                width = cli_width
            elif cli_resolution is not None:
                width = ALL_RESOLUTION_PRESETS[cli_resolution][0]
            else:
                width = config_width if config_width is not None else 512

            if cli_height is not None:
                height = cli_height
            elif cli_resolution is not None:
                height = ALL_RESOLUTION_PRESETS[cli_resolution][1]
            else:
                height = config_height if config_height is not None else 512
        elif cli_resolution is not None:
            # Explicit --resolution overrides config file
            width, height = ALL_RESOLUTION_PRESETS[cli_resolution]
        else:
            # No resolution flag: fall back to config, then default
            width = config_width if config_width is not None else 512
            height = config_height if config_height is not None else 512

        # Pre-load model before timer starts
        from fluxgen.generator import ModelManager
        preloaded_model = ModelManager.get_model(
            model_name=args.model,
            quantize=preset.get("quantize"),
        )

        start = time.perf_counter() if args.timer else None
        generate_image(
            prompt=args.prompt,
            preset=preset,
            seed=args.seed,
            output=output_path,
            width=width,
            height=height,
            style=args.style,
            custom_styles=config.get("styles"),
            init_image=args.init_image,
            strength=args.strength,
            model_name=args.model,
            model=preloaded_model,
        )
        if start is not None:
            elapsed = time.perf_counter() - start
            logger.info(f"\u23a1 Generated in {elapsed:.2f}s")

def handle_edit(args, interactive=False):
    with error_handler(args, interactive):
        from fluxgen.editor import ImageEditor, EDIT_DEFAULT_STEPS

        # Robust path resolution (handles ~ and relative paths)
        input_paths = [Path(img).expanduser().resolve() for img in args.image]
        for p in input_paths:
            if not p.exists():
                raise FileNotFoundError(f"Input image not found: {p}")
            if not p.is_file():
                raise ValueError(f"Input image must be a file: {p}")

        def generate_edit_filename():
            base_name = input_paths[0].stem
            try:
                from wonderwords import RandomWord
                rw = RandomWord()
                random_word = rw.random_words(1, word_max_length=5)[0]
                return f"{base_name}_{random_word}.png"
            except ImportError:
                from fluxgen.generator import generate_random_filename
                return generate_random_filename()

        output_path = resolve_output_path(args.output, args.output_dir, generate_edit_filename)

        model_name = getattr(args, "model", "flux2-klein")
        quantize = getattr(args, "quantize", None)
        seed = getattr(args, "seed", None)

        logger.info(f"Initializing ImageEditor with model '{model_name}'...")
        editor = ImageEditor(model_name=model_name, quantize=quantize)
 
        # Pre-load model before timer starts
        if model_name == "qwen-image-edit":
            logger.info("First run will download the Q4_K_M GGUF model (~13GB). It requires disk space and a stable connection.")
            logger.info("Loading model weights...")
        else:
            logger.info(f"Loading mflux model '{model_name}' weights...")
        editor._load_pipeline()

        logger.info(f"Applying edit: '{args.prompt}'")
        start = time.perf_counter() if args.timer else None
        editor.edit(
            image_paths=args.image,
            prompt=args.prompt,
            output_path=output_path,
            steps=args.steps,
            guidance_scale=args.guidance,
            seed=seed,
            width=args.width,
            height=args.height,
        )

        if start is not None:
            elapsed = time.perf_counter() - start
            logger.info(f"\u23a1 Edited in {elapsed:.2f}s")

if __name__ == "__main__":
    main()
