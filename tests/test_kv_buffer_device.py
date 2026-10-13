"""kv_transfer.kv_buffer_device: cuda (default) or cpu (NIXL host staging buffer)."""

import pytest

from canatune.config import ConfigError, kv_buffer_device, load_config


def test_default_is_cuda_and_cpu_is_accepted() -> None:
    config = load_config()
    assert kv_buffer_device(config) == "cuda"
    config["kv_transfer"]["kv_buffer_device"] = "cpu"
    assert kv_buffer_device(config) == "cpu"


def test_other_devices_are_refused() -> None:
    config = load_config()
    config["kv_transfer"]["kv_buffer_device"] = "xpu"
    with pytest.raises(ConfigError):
        kv_buffer_device(config)
