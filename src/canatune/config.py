"""Project configuration loading."""

from collections.abc import Mapping
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


NIXL_CONNECTOR = "NixlConnector"
P2P_NCCL_CONNECTOR = "P2pNcclConnector"


def kv_connector(config: Mapping[str, Any]) -> str:
    """kv_transfer.connector: NixlConnector (default; D pulls P's KV into blocks it has
    allocated) or P2pNcclConnector (P pushes into D's receive buffer)."""
    value = (config.get("kv_transfer") or {}).get("connector", NIXL_CONNECTOR)
    if value not in (NIXL_CONNECTOR, P2P_NCCL_CONNECTOR):
        raise ConfigError(
            f"kv_transfer.connector must be {NIXL_CONNECTOR} or {P2P_NCCL_CONNECTOR}"
        )
    return value


def kv_receive_buffer_bytes(config: Mapping[str, Any]) -> float | None:
    """D's KV receive buffer in bytes: P2pNcclConnector's kv_buffer_size
    (kv_transfer.kv_buffer_bytes, vLLM default 1e9). None for NixlConnector: D reads
    the KV straight into its paged KV cache, so there is no buffer to overflow and no
    in-flight KV gate."""
    if kv_connector(config) == NIXL_CONNECTOR:
        return None
    value = (config.get("kv_transfer") or {}).get("kv_buffer_bytes", 1e9)
    try:
        number = float(value)  # YAML 1.1 reads 1.0e9 as a string
    except (TypeError, ValueError):
        raise ConfigError("kv_transfer.kv_buffer_bytes must be a positive number") from None
    if not number > 0:
        raise ConfigError("kv_transfer.kv_buffer_bytes must be a positive number")
    return number
