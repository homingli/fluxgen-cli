"""fluxgen.cli — top-level CLI entry point.

Package layout (introduced when ``cli.py`` exceeded ~480 lines):

- :mod:`fluxgen.cli.presets_arg` — token-shape constants, argv
  normalization, and the small argparse builders (``add_verbosity_flags``,
  ``add_preset_args``, ``add_resolution_args``).
- :mod:`fluxgen.cli.commands` — ``generate`` and ``edit`` subparser
  definitions, their handlers, the output-path resolver, the
  error-handling context manager, and the resolution priority chain
  (:func:`fluxgen.cli.commands.resolve_image_dimensions`).
- :mod:`fluxgen.cli.interactive` — the REPL parser subclass and the
  ``handle_interactive`` REPL loop.

This module owns the things that tie everything together: the
version resolver (cached, with ``importlib.metadata`` + ``pyproject.toml``
fallback), ``setup_logging``, ``suppress_external_output``, the
``get_parser`` aggregator that wires the subparsers from each
sub-module into a single root parser, and ``main`` (the entry point
declared in ``pyproject.toml``'s ``[project.scripts]``).

Re-exports: ``handle_generate`` and ``handle_edit`` are re-exported
so that test fixtures (and any future embedders) that import
``fluxgen.cli`` see the full public surface without having to know
the internal split. ``load_config`` and ``distribution`` are also
re-exported because the existing test suite patches them via
``patch.object(cli, "load_config")`` /
``patch.object(cli, "distribution")``; those patches only work when
the names live on this module's namespace.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path

try:
    from importlib.metadata import distribution
except ImportError:
    from importlib_metadata import distribution  # Python < 3.8 fallback

from fluxgen.cli.commands import (
    handle_edit,
    handle_generate,
)
from fluxgen.cli.interactive import handle_interactive
from fluxgen.cli.presets_arg import (
    COMMANDS,
    GLOBAL_FLAGS,
    PASSTHROUGH_FLAGS,
    _resolve_log_level_and_fmt,
    add_verbosity_flags,
    with_default_command,
)
from fluxgen.config import load_config


logger = logging.getLogger("fluxgen")


# ── Logging setup ───────────────────────────────────────────────────────────


def setup_logging(verbose=False, silent=False):
    """Install a single ``StreamHandler`` on the ``fluxgen`` logger.

    Clears any previously-attached handlers first so repeated calls
    (e.g. once in ``main()`` and again implicitly by callers that
    re-set verbosity) don't stack duplicate handlers. Sets
    ``propagate=False`` so the root logger doesn't double-print our
    messages.

    The level/format mapping lives in
    :func:`fluxgen.cli.presets_arg._resolve_log_level_and_fmt`.
    """
    level, fmt = _resolve_log_level_and_fmt(verbose, silent)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt))
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


# ── Output redirection ─────────────────────────────────────────────────────


@contextmanager
def suppress_external_output(enabled):
    """Redirect stdout/stderr to ``/dev/null`` for the duration of the block.

    Used while ``--silent`` is set so that progress bars / status
    lines emitted by underlying libraries (mflux, diffusers,
    huggingface_hub) don't bypass our own logger filter. When
    ``enabled`` is False, the context manager is a no-op (via
    :class:`contextlib.nullcontext`).
    """
    if not enabled:
        with nullcontext():
            yield
        return

    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


# ── Version resolution ──────────────────────────────────────────────────────

_cached_version: str | None = None


def _get_version() -> str:
    """Resolve the fluxgen-cli version. Cached after first call.

    Looks up via :func:`importlib.metadata.distribution` first;
    falls back to reading ``pyproject.toml`` directly when the
    package is not installed (e.g. running from a source checkout
    without ``uv sync``). Returns ``"unknown"`` if neither path
    yields a version.

    The cache lives in :data:`_cached_version` so callers can reset
    it (the test suite does this to verify the lookup paths in
    isolation).
    """
    global _cached_version
    if _cached_version is not None:
        return _cached_version

    try:
        _cached_version = distribution("fluxgen-cli").version
        return _cached_version
    except (ImportError, FileNotFoundError):
        pass

    try:
        import tomllib
        # When this module lives at ``fluxgen/cli/__init__.py`` (the
        # post-split layout), ``parent.parent.parent`` walks back up
        # to the repo root where ``pyproject.toml`` sits. The
        # pre-split flat module was at ``fluxgen/cli.py``, so
        # ``parent.parent`` sufficed there — the extra ``parent`` here
        # accounts for the new ``cli/`` package directory.
        with Path(__file__).parent.parent.parent.joinpath("pyproject.toml").open("rb") as f:
            _cached_version = tomllib.load(f)["project"]["version"]
            return _cached_version
    except (ImportError, KeyError, OSError):
        pass

    _cached_version = "unknown"
    return _cached_version


# ── Parser aggregator ──────────────────────────────────────────────────────


def get_parser(config, version, interactive=False):
    """Build the root argument parser and attach every subparser.

    When ``interactive=True``, returns an :class:`InteractiveParser`
    that converts ``sys.exit`` into :class:`ParserExit` so the REPL
    loop can survive ``--help`` / parse errors. Otherwise uses the
    stock :class:`argparse.ArgumentParser`, which exits the process
    on ``--help`` / ``--version`` (the desired behavior for the
    one-shot CLI).

    Subparsers are built by delegating to
    :func:`fluxgen.cli.commands.add_generate_parser`,
    :func:`fluxgen.cli.commands.add_edit_parser`, and the inline
    interactive parser block. The shared verbosity parent (built
    here via :func:`fluxgen.cli.presets_arg.add_verbosity_flags`) is
    reused as ``parents=[verbosity_parent]`` on every subparser so
    ``-v/--verbose`` / ``-s/--silent`` behave identically and the
    help text stays in one place.
    """
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

    # Lazy import to keep `get_parser` decoupled from the heavy
    # `commands` module's transitive imports (mflux, diffusers).
    # The subparser builders themselves only touch argparse and
    # `fluxgen.config` / `fluxgen.presets`, so this stays cheap.
    from fluxgen.cli.commands import add_edit_parser, add_generate_parser

    add_generate_parser(subparsers, verbosity_parent, config)
    add_edit_parser(subparsers, verbosity_parent, config)

    # Interactive subparser is a one-liner — kept inline rather than
    # delegated to interactive.py so it doesn't import REPL-only
    # dependencies (readline, shlex) at parser-build time.
    subparsers.add_parser(
        "interactive",
        aliases=["repl"],
        help="Start an interactive session to keep models loaded in memory",
        parents=[verbosity_parent],
    )

    return parser


# ── Entry point ────────────────────────────────────────────────────────────


def main(argv=None):
    """Top-level CLI entry, wired in ``pyproject.toml`` as ``fluxgen``.

    Flow:

    1. Normalize argv via :func:`with_default_command` so
       ``fluxgen "prompt"`` becomes ``fluxgen generate "prompt"``.
    2. Detect passthrough flags (``--help``, ``--version``) and
       skip config loading on that fast path — argparse will exit
       before any handler runs.
    3. Parse argv via :func:`get_parser`.
    4. Reject ``--silent`` + ``--verbose`` (mutually exclusive).
    5. Install the logger handler.
    6. Dispatch to ``handle_generate`` / ``handle_edit`` /
       ``handle_interactive`` under
       :func:`suppress_external_output` so ``--silent`` actually
       silences underlying libraries.
    """
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    adjusted_argv = with_default_command(raw_argv)

    # Fast path: --help / --version are handled by argparse natively
    # and exit before handle_* ever runs, so they don't need
    # config-driven defaults. Skip the config load (and the cached
    # version lookup if it's the very first call) for these.
    is_passthrough = any(arg in PASSTHROUGH_FLAGS for arg in adjusted_argv)
    config = {} if is_passthrough else load_config()

    parser = get_parser(config, _get_version())
    args = parser.parse_args(adjusted_argv)

    if getattr(args, "verbose", False) and getattr(args, "silent", False):
        parser.error("argument -s/--silent: not allowed with argument -v/--verbose")

    # Resolve verbosity/silent and apply logging
    silent = getattr(args, "silent", False)
    setup_logging(verbose=getattr(args, "verbose", False), silent=silent)

    with suppress_external_output(silent):
        if args.command in ["generate", "gen"]:
            handle_generate(args, config)
        elif args.command == "edit":
            handle_edit(args, config=config)
        elif args.command in ["interactive", "repl"]:
            handle_interactive(config, _get_version())
        else:
            parser.print_help()


__all__ = [
    "COMMANDS",
    "GLOBAL_FLAGS",
    "PASSTHROUGH_FLAGS",
    "_cached_version",
    "_get_version",
    "_resolve_log_level_and_fmt",
    "add_verbosity_flags",
    "get_parser",
    "handle_edit",
    "handle_generate",
    "handle_interactive",
    "load_config",
    "main",
    "setup_logging",
    "suppress_external_output",
    "with_default_command",
]


if __name__ == "__main__":
    main()
