"""Argument-building helpers and argv manipulation for the CLI.

Centralizes the constants and helpers that govern how command-line
arguments are tokenized, normalized, and registered on subparsers.
Lives in its own module so that ``cli/__init__.py`` (which owns
``get_parser`` and ``main``) doesn't have to inline dozens of
argparse flag definitions.

Scope:

- Token-shape constants (``GLOBAL_FLAGS``, ``COMMANDS``,
  ``PASSTHROUGH_FLAGS``) used by both ``with_default_command``
  (argv normalization) and the interactive REPL (command detection).
- ``with_default_command``: insert ``generate`` before the first
  non-global token when the user omitted an explicit subcommand,
  e.g. ``fluxgen "a prompt"`` -> ``fluxgen generate "a prompt"``.
- ``add_verbosity_flags`` / ``add_preset_args`` / ``add_resolution_args``:
  reusable argparse builders shared by the ``generate`` and ``edit``
  subparsers.
- ``_resolve_log_level_and_fmt``: maps a ``(verbose, silent)`` pair
  (the parsed values of the verbosity flags) to a logging level +
  format string. Used by ``setup_logging`` (in ``cli/__init__.py``)
  and by the REPL's per-command log reconfiguration.
"""

from __future__ import annotations

import argparse
import logging

from fluxgen.presets import ALL_RESOLUTION_PRESETS


# ── argv shape constants ────────────────────────────────────────────────────

# Flags that may appear *anywhere* before the subcommand (e.g.
# `fluxgen -s gen "..."` or `fluxgen gen -v "..."`). `with_default_command`
# skips past these when deciding where to insert the default `generate`.
GLOBAL_FLAGS = {"-v", "--verbose", "-s", "--silent"}

# Subcommands recognized by both `main()` and the interactive REPL.
# `gen` aliases `generate`; `repl` aliases `interactive`.
COMMANDS = {"generate", "gen", "edit", "interactive", "repl"}

# Flags that argparse handles natively (so we exit before any
# handle_* runs and skip config loading on the fast path).
PASSTHROUGH_FLAGS = {"--version", "--help", "-h"}


def with_default_command(argv):
    """Insert `generate` before the first non-global token when no
    subcommand is present.

    Lets ``fluxgen "a prompt"`` mean ``fluxgen generate "a prompt"``
    without losing global flags (``-v`` / ``-s``) that come before
    the prompt. If the first non-global token is already a known
    subcommand or a passthrough flag (``--help`` / ``--version``),
    argv is returned unchanged.
    """
    insert_at = 0
    while insert_at < len(argv) and argv[insert_at] in GLOBAL_FLAGS:
        insert_at += 1

    if insert_at == len(argv):
        return argv

    token = argv[insert_at]
    if token in COMMANDS or token in PASSTHROUGH_FLAGS:
        return argv

    return argv[:insert_at] + ["generate"] + argv[insert_at:]


# ── argparse builders ───────────────────────────────────────────────────────


def add_verbosity_flags(parser):
    """Attach mutually-exclusive ``-v/--verbose`` and ``-s/--silent`` flags.

    Uses ``argparse.SUPPRESS`` as the default so the flag is absent
    from the parsed namespace when the user didn't pass one. This
    lets ``main()`` detect "neither flag set" without conflicting
    with argparse's normal ``False`` default and lets the verbosity
    flags be attached to a shared parent parser that's reused across
    every subparser (so the help text doesn't drift).
    """
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


def add_preset_args(parser):
    """Attach the preset-selection flags (``-0`` / ``-3`` / ``-8`` /
    ``--preset``) to a subparser.

    All four flags share ``dest="preset_idx"``: the numeric shortcuts
    store an integer index (``0``/``3``/``8``); ``--preset NAME``
    stores a name that ``handle_generate`` resolves via
    ``PRESETS_BY_NAME``. Because argparse applies later flags after
    earlier ones, ``--preset`` overrides any numeric shortcut when
    both are passed.
    """
    parser.add_argument(
        "-0", "--fast", action="store_const", const=0, dest="preset_idx",
        help="Fast preset (default)",
    )
    parser.add_argument(
        "-3", "--standard", action="store_const", const=3, dest="preset_idx",
        help="Standard preset",
    )
    parser.add_argument(
        "-8", "--quality", action="store_const", const=8, dest="preset_idx",
        help="Quality preset",
    )
    parser.add_argument(
        "--preset", choices=["fast", "standard", "quality"],
        help="Named preset (overrides numeric flags)",
    )


def add_resolution_args(parser):
    """Attach ``--resolution`` / ``--width`` / ``--height`` flags.

    All three default to ``argparse.SUPPRESS`` so that the priority
    chain in :func:`fluxgen.cli.commands.resolve_image_dimensions`
    can distinguish "user didn't pass anything" (attribute absent)
    from "user passed an explicit value" (attribute present). This
    is what enables partial-axis fallback: e.g. ``--width 800`` alone
    falls back through ``--resolution`` > config > 512 for the height
    axis rather than blindly defaulting.
    """
    parser.add_argument(
        "--resolution", "-r",
        type=str,
        choices=list(ALL_RESOLUTION_PRESETS.keys()),
        default=argparse.SUPPRESS,
        help="Resolution preset (default: tiny 512x512 for faster generation)",
    )
    parser.add_argument(
        "--width", type=int, default=argparse.SUPPRESS,
        help="Image width (overrides --resolution)",
    )
    parser.add_argument(
        "--height", type=int, default=argparse.SUPPRESS,
        help="Image height (overrides --resolution)",
    )


# ── logging level/format mapping ────────────────────────────────────────────


def _resolve_log_level_and_fmt(verbose, silent):
    """Map a ``(verbose, silent)`` flag pair to ``(logging level, format)``.

    Verbose wins over silent when both are set — but in practice
    ``main()`` rejects that combination via ``parser.error()`` before
    this is ever called, so the silent-fallback branch is defensive.
    The format string drops the level prefix in normal mode so
    single-line progress messages read cleanly; verbose mode keeps
    the level so debugging output is grep-able.
    """
    level = logging.DEBUG if verbose else logging.ERROR if silent else logging.INFO
    fmt = "%(levelname)s: %(message)s" if verbose else "%(message)s"
    return level, fmt
