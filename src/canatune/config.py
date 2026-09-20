"""Project configuration loading."""

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file cannot be loaded."""


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load a YAML configuration file as a mapping.

    Variable expressions such as ``${work_dir}`` are preserved for a later
    configuration-resolution stage.
    """
    config_path = Path(path)

    try:
        content = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"Unable to read configuration: {config_path}") from error

    try:
        config = yaml.safe_load(content)
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML configuration: {config_path}") from error

    if not isinstance(config, dict):
        raise ConfigError("The configuration root must be a mapping")

    return config
