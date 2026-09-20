"""Command-line entry point for the CanaTune service."""

import uvicorn


def main() -> None:
    """Run the development ASGI server."""
    uvicorn.run("canatune.app:create_app", factory=True)


if __name__ == "__main__":
    main()
