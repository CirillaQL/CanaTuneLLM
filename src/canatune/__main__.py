"""Command-line entry point for the CanaTune service."""

import os

import uvicorn

from canatune.config import load_config
from canatune.infrastructure.stack_sampler import start_from_env


def main() -> None:
    """Run the configured ASGI server."""
    config = load_config(os.environ.get("CANATUNE_CONFIG", "config.yaml"))
    proxy = config["proxy"]
    start_from_env()  # CANATUNE_PROFILE_OUT: stack samples of the event loop thread
    from canatune.proxy.workers import proxy_workers, run_decision

    workers = proxy_workers(config)
    if workers > 0:  # the decision process; N processes relay the requests
        run_decision(config, workers)
        return
    uvicorn.run(
        "canatune.app:create_app",
        factory=True,
        host=str(proxy["host"]),
        port=int(proxy["port"]),
        access_log=False,  # per-request lines would flood the job console
    )


if __name__ == "__main__":
    main()
