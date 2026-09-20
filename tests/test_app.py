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
