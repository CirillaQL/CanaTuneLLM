"""Command-line entry point for the CanaTune service."""

import os

import uvicorn

from canatune.config import load_config


def main() -> None:
    """Run the configured ASGI server."""
    config = load_config(os.environ.get("CANATUNE_CONFIG", "config.yaml"))
    proxy = config["proxy"]
    uvicorn.run(
        "canatune.app:create_app",
        factory=True,
        host=str(proxy["host"]),
        port=int(proxy["port"]),
    )


if __name__ == "__main__":
    main()
