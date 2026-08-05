import logging
import tomllib
from pathlib import Path
from typing import Any

logger = logging.getLogger("fluxgen")

DEFAULT_CONFIG_FILENAME = ".fluxgen.toml"

# Known keys under [defaults]. Unknown keys are logged + dropped so
# typos in `.fluxgen.toml` surface immediately instead of being silently
# ignored. Add new user-facing defaults here when they are introduced.
_KNOWN_DEFAULT_KEYS: frozenset[str] = frozenset({
    "model",
    "output_dir",
    "width",
    "height",
    "preset",
    "style",
    "max_edit_dimension",
})


def load_config() -> dict[str, Any]:
    """Load configuration from .fluxgen.toml in home dir and current dir.

    Returns a dict with optional 'defaults' and 'styles' sections. The two
    locations are processed in order: home first, then current directory.
    dict.update() is later-wins, so current-directory values override home
    values when both files exist.

    Unknown keys under ``[defaults]`` are logged at WARNING and dropped
    so they cannot silently override CLI behavior on typos. ``[styles]``
    is treated as open-ended (user-defined style suffixes), so its keys
    are not validated.
    """
    config: dict[str, Any] = {}

    # Process in order: home first, then cwd. dict.update() is later-wins,
    # so cwd overrides home when both files exist.
    locations = [
        Path.home() / DEFAULT_CONFIG_FILENAME,
        Path.cwd() / DEFAULT_CONFIG_FILENAME,
    ]

    for loc in locations:
        if loc.exists():
            try:
                with open(loc, "rb") as f:
                    data = tomllib.load(f)
                    if "defaults" in data:
                        if "defaults" not in config:
                            config["defaults"] = {}
                        merged = config["defaults"]
                        for k, v in data["defaults"].items():
                            if k not in _KNOWN_DEFAULT_KEYS:
                                logger.warning(
                                    f"Ignoring unknown key '{k}' in [defaults] of {loc}. "
                                    f"Known keys: {', '.join(sorted(_KNOWN_DEFAULT_KEYS))}."
                                )
                                continue
                            merged[k] = v
                    if "styles" in data:
                        if "styles" not in config:
                            config["styles"] = {}
                        config["styles"].update(data["styles"])
            except (tomllib.TOMLDecodeError, OSError) as e:
                logger.warning(f"Failed to load config from {loc}: {e}")

    return config

def get_config_value(config: dict[str, Any], key: str, default: Any = None) -> Any:
    """Helper to get a value from the defaults section of the config."""
    return config.get("defaults", {}).get(key, default)
