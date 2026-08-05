"""Interactive REPL mode and the parser used inside it.

The REPL keeps a single model instance warm across commands by
looping on ``input()`` rather than re-spawning a Python process.
That changes three things relative to the one-shot CLI path:

- :class:`InteractiveParser` swaps out argparse's default
  ``sys.exit`` on ``--help`` / parse errors for
  :class:`ParserExit` so the REPL loop can catch the exit and
  continue. Without this, ``fluxgen> help`` would terminate the
  session.
- Verbosity flags apply per-command, not per-session: each new
  command re-applies ``setup_logging``'s level/format to the
  shared ``fluxgen`` logger so the previous command's verbose
  setting doesn't leak into the next.
- :class:`ParserExit` and :class:`InteractiveParser` are exported
  for test fixtures that want to drive the REPL programmatically.

``handle_interactive`` does its CLI-package imports lazily (see the
function body) so this module is cheap to import on its own and so
that any future top-level dependency the REPL takes on
``fluxgen.cli`` doesn't form a load-time cycle. There's no cycle
today — ``__init__.py`` only re-exports ``handle_interactive`` from
this module, and this module's top-level imports go one way
(``presets_arg`` only) — but deferring keeps that property stable
and also avoids pulling ``argparse`` / ``sys`` work into the
parser-build path that ``--help`` and ``--version`` already pay
for.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import sys

from fluxgen.cli.presets_arg import COMMANDS


# Tokens that exit the REPL. Module-level constant so the comparison
# is one hash-lookup per keystroke instead of re-allocating a list
# every iteration.
_EXIT_TOKENS = frozenset({"exit", "quit"})


logger = logging.getLogger("fluxgen")


class ParserExit(Exception):
    """Raised in place of ``sys.exit`` when an REPL-internal parser
    encounters ``--help`` or a parse error.

    The REPL loop catches this and continues, instead of letting
    the exception bubble out and kill the session.
    """


class InteractiveParser(argparse.ArgumentParser):
    """Argparse subclass that converts ``exit`` into a raiseable
    :class:`ParserExit`.

    Without this override, ``parser.parse_args(['--help'])`` inside
    the REPL would call ``sys.exit(0)`` and terminate the whole
    session. The override routes both ``exit()`` and ``error()``
    through :class:`ParserExit` so the loop can catch them.
    """

    def exit(self, status=0, message=None):
        if message:
            self._print_message(message, sys.stderr)
        raise ParserExit()

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n")


# ASCII-art banner shown once at the top of the REPL. Kept as a
# module-level constant (not inside handle_interactive) so the
# raw string doesn't pollute the function body and tests can
# grep for the banner without parsing source.
_BANNER = r"""
  __ _
 / _| |
| |_| |_   ___  ____ _  ___ _ __
|  _| | | | \ \/ / _` |/ _ \ '_ \
| | | | |_| |>  < (_| |  __/ | | |
|_| |_|\__,_/_/\_\__, |\___|_| |_|
                  __/ |
                 |___/
"""


def add_interactive_parser(subparsers, verbosity_parent):
    """Attach the ``interactive`` / ``repl`` subparser to ``subparsers``.

    Lifted out of :func:`fluxgen.cli.get_parser` so the parser-build
    path is symmetric across subcommands — each subcommand's
    definition lives in its handler module. The body itself is just
    argparse glue: name, alias, help text, and the shared verbosity
    parent. The REPL loop (:func:`handle_interactive`) is the only
    REPL-only consumer and is invoked at runtime, not at parser-build.
    """
    subparsers.add_parser(
        "interactive",
        aliases=["repl"],
        help="Start an interactive session to keep models loaded in memory",
        parents=[verbosity_parent],
    )


def handle_interactive(config, version):
    """Drive the interactive REPL.

    The loop:

    1. Reads a line, parses it with ``shlex`` (so quoted prompts
       don't get split into separate "tokens"), and dispatches on
       the first token.
    2. ``help`` / ``-h`` / ``--help`` prints the parser help and
       continues (without consuming a model).
    3. Empty input, ``exit``, ``quit``, ``KeyboardInterrupt``, and
       ``EOFError`` all exit cleanly.
    4. Unknown first tokens are silently ignored — the REPL is
       intentionally lenient about typos. This matches the
       pre-split behavior.
    5. Each command goes through the shared :mod:`fluxgen.cli`
       handlers (``handle_generate`` / ``handle_edit``) with
       ``interactive=True`` so they don't call ``sys.exit(1)`` on
       error; failures instead fall back to the loop's outer
       ``except Exception`` which logs and continues.

    The ``version`` arg is passed in (rather than re-resolved here)
    so the version cache survives the REPL process lifetime and
    matches the ``--version`` output from a non-REPL invocation.
    """
    # Imported lazily so the REPL module stays cheap to import on its
    # own and so the load-time cycle stays one-way (see module
    # docstring). Deferring also keeps the parser-build and import
    # surface clean: ``get_parser`` is only needed once the user has
    # actually entered the REPL.
    from fluxgen.cli import get_parser, setup_logging, suppress_external_output
    from fluxgen.cli.commands import handle_edit, handle_generate
    from fluxgen.cli.presets_arg import resolve_log_level_and_fmt

    try:
        import readline  # noqa: F401  -- imported for side effect of enabling line editing / history
    except ImportError:
        pass

    print(_BANNER)
    logger.info(
        "Starting fluxgen interactive mode. Type 'exit', 'quit' or 'help' to navigate."
    )
    parser = get_parser(config, version, interactive=True)

    while True:
        try:
            cmd = input("\nfluxgen> ").strip()
            if not cmd:
                continue
            if cmd.lower() in _EXIT_TOKENS:
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
            verbose = getattr(args, "verbose", False)
            level, fmt = resolve_log_level_and_fmt(verbose, silent)
            # ``setup_logging`` was already called once by ``main()``
            # before the REPL started, so a handler is installed. The
            # defensive reinstall below only fires if a future entry
            # path bypasses ``main()``; in the common case this is just
            # a re-aim at the current command's verbosity.
            if not logger.handlers:
                setup_logging(verbose=verbose, silent=silent)
            handler = logger.handlers[0]
            handler.setLevel(level)
            handler.setFormatter(logging.Formatter(fmt))
            logger.setLevel(level)

            with suppress_external_output(silent):
                if args.command in ["generate", "gen"]:
                    handle_generate(args, config, interactive=True)
                elif args.command == "edit":
                    handle_edit(args, config=config, interactive=True)
                elif args.command in ["interactive", "repl"]:
                    logger.error("Already in interactive mode.")
                else:
                    parser.print_help()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        except Exception as e:
            logger.error(f"Error: {e}")
