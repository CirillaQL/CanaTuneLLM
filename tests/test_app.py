"""Smoke tests for the application skeleton."""

from canatune.app import create_app
from canatune.config import load_config


def test_create_app() -> None:
    app = create_app()

    assert app.title == "CanaTune"
    assert app.version == "0.1.0"


def test_load_project_config() -> None:
    config = load_config()

    assert config["project"]["mode"] == "disaggregated_inference"
    assert config["topology"]["endpoints"]["P0"]["role"] == "prefill"


def test_cold_start_probes_generate_the_service_output_cap() -> None:
    import pytest

    from canatune.service import cold_start_lengths

    config = load_config()
    config["canary"]["max_output_tokens"] = 512
    config["canary"]["default_prompts"] = [256, 1024]
    assert cold_start_lengths(config) == [(256, 512), (1024, 512)]
    # Without the cap the legacy explicit pairs still work.
    del config["canary"]["max_output_tokens"]
    config["canary"]["default_lengths"] = [[100, 20], [300, 40]]
    assert cold_start_lengths(config) == [(100, 20), (300, 40)]
    # The cap with legacy pairs keeps their prompts.
    config["canary"]["max_output_tokens"] = 128
    del config["canary"]["default_prompts"]
    assert cold_start_lengths(config) == [(100, 128), (300, 128)]
    for bad in (0, -1, 1.5, True):
        config["canary"]["max_output_tokens"] = bad
        with pytest.raises(ValueError):
            cold_start_lengths(config)
    config["canary"]["max_output_tokens"] = 3000
    config["canary"]["default_prompts"] = [2048]  # 2048 + 3000 > max_model_len 4096
    with pytest.raises(ValueError, match="max_model_len"):
        cold_start_lengths(config)


def test_runtime_lengths_start_from_the_output_cap() -> None:
    import httpx

    from canatune.service import build_runtime

    config = load_config()
    config["routing"]["policy"] = "cantune"
    config["telemetry"]["enabled"] = False
    config["canary"]["max_output_tokens"] = 256
    runtime = build_runtime(config, {}, lambda: httpx.AsyncClient())
    assert runtime.lengths.using_default
    assert {o for _, o in runtime.lengths.pairs()} == {256}
