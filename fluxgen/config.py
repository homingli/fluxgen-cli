import tomllib
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_FILENAME = ".fluxgen.toml"

def load_config() -> dict[str, Any]:
    """Load configuration from .fluxgen.toml in current dir or home dir."""
    config = {}
    
    # Locations to check (in order of priority: local then home)
    locations = [
        Path.home() / DEFAULT_CONFIG_FILENAME,
        Path.cwd() / DEFAULT_CONFIG_FILENAME,
    ]
    
    for loc in locations:
        if loc.exists():
            try:
                with open(loc, "rb") as f:
                    data = tomllib.load(f)
                    # Deep merge or update? Let's do a simple update for now
                    if "defaults" in data:
                        if "defaults" not in config:
                            config["defaults"] = {}
                        config["defaults"].update(data["defaults"])
                    if "styles" in data:
                        if "styles" not in config:
                            config["styles"] = {}
                        config["styles"].update(data["styles"])
            except Exception as e:
                print(f"Warning: Failed to load config from {loc}: {e}")
                
    return config

def get_config_value(config: dict[str, Any], key: str, default: Any = None) -> Any:
    """Helper to get a value from the defaults section of the config."""
    return config.get("defaults", {}).get(key, default)
