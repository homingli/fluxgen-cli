"""fluxgen.cli — top-level CLI entry point.

Package layout (introduced when the monolithic ``cli.py`` was split
during a cleanup pass):

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
    add_edit_parser,
    add_generate_parser,
    handle_edit,
    handle_generate,
)
from fluxgen.cli.interactive import (
    add_interactive_parser,
    handle_interactive,
)
from fluxgen.cli.presets_arg import (
    COMMANDS,
    GLOBAL_FLAGS,
    PASSTHROUGH_FLAGS,
    add_verbosity_flags,
    resolve_log_level_and_fmt,
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
    level, fmt = resolve_log_level_and_fmt(verbose, silent)
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
    delegates to :data:`fluxgen.__version__` when that fails (which
    itself has a hard-coded literal fallback kept in sync with
    ``pyproject.toml`` via
    :func:`test_fluxgen_fallback_literal_matches_pyproject`).

    This keeps ``_get_version`` and ``fluxgen.__version__`` in
    lockstep — a single source of truth — rather than each maintaining
    its own independent fallback chain (which previously diverged:
    CLI's chain included a ``pyproject.toml`` read and an ``"unknown"``
    sentinel, while ``fluxgen.__version__`` had a hard-coded literal).

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

    # Lazy import to avoid a circular import: ``fluxgen`` is the
    # top-level package and importing it at module load would pull
    # in the entire CLI / generator / editor graph before this
    # module finishes initializing.
    from fluxgen import __version__ as _pkg_version
    _cached_version = _pkg_version
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

    Subparsers are built by delegating to each handler module's
    ``add_*_parser`` function. The shared verbosity parent (built
    here via :func:`fluxgen.cli.presets_arg.add_verbosity_flags`) is
    reused as ``parents=[verbosity_parent]`` on every subparser so
    ``-v/--verbose`` / ``-s/--silent`` behave identically and the
    help text stays in one place.

    Note on import cost: ``commands`` is imported at the top of this
    module for ``handle_generate`` / ``handle_edit`` re-exports, so
    ``fluxgen.editor`` (diffusers, huggingface_hub) and
    ``fluxgen.generator` (mflux) are pulled in at parser-build time
    even for ``--help`` / ``--version``. The pre-split single-module
    layout paid the same cost; the post-split package keeps it. The
    passthrough-flag fast path (``main()``) skips ``load_config`` on
    ``--help`` / ``--version`` but cannot avoid the import-time
    transitive cost without a more invasive lazy-import refactor
    that would also break the existing ``patch.object(cli, X)`` test
    contracts.
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

    add_generate_parser(subparsers, verbosity_parent, config)
    add_edit_parser(subparsers, verbosity_parent, config)
    add_interactive_parser(subparsers, verbosity_parent)

    return parser


# ── Entry point ────────────────────────────────────────────────────────────


# ── Subcommand dispatch ──────────────────────────────────────────────────
#
# Maps each subcommand token (canonical name + alias) to its handler.
# The dispatch in :func:`main` is a single ``.get()`` lookup instead
# of an if/elif chain — keeps the alias list in one place rather than
# scattered as ``["generate", "gen"]`` literals, and the membership
# check below proves all dispatch targets are real commands (so the
# table stays in sync with :data:`fluxgen.cli.presets_arg.COMMANDS`).

def _dispatch_generate(args, config, _version):
    handle_generate(args, config)


def _dispatch_edit(args, config, _version):
    handle_edit(args, config=config)


def _dispatch_interactive(args, config, version):
    handle_interactive(config, version)


_DISPATCH = {
    "generate": _dispatch_generate,
    "gen": _dispatch_generate,
    "edit": _dispatch_edit,
    "interactive": _dispatch_interactive,
    "repl": _dispatch_interactive,
}


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
    6. Dispatch by subcommand via :data:`_DISPATCH` under
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

    handler = _DISPATCH.get(args.command)
    if handler is None:
        parser.print_help()
        return

    with suppress_external_output(silent):
        handler(args, config, _get_version())


__all__ = [
    "COMMANDS",
    "GLOBAL_FLAGS",
    "PASSTHROUGH_FLAGS",
    "_cached_version",
    "_get_version",
    "add_verbosity_flags",
    "distribution",
    "get_parser",
    "handle_edit",
    "handle_generate",
    "handle_interactive",
    "load_config",
    "main",
    "resolve_log_level_and_fmt",
    "setup_logging",
    "suppress_external_output",
    "with_default_command",
]


if __name__ == "__main__":
    main()
