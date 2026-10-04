"""``python -m src.mcp_server``: run Finnie's MCP server.

- Default: stdio, for Claude Desktop (which starts the process itself). stdout carries
  the protocol, so logs go to stderr.
- ``--http [--port N]``: Streamable HTTP on 127.0.0.1, protected by MCP_API_TOKEN.

Logs never contain tool arguments, results, or the token.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

from src.core.config import PROJECT_ROOT, ConfigError, Settings, get_settings
from src.utils.logging import configure_logging

if TYPE_CHECKING:
    from mcp.server import MCPServer


def main(argv: list[str] | None = None) -> int:
    """Run the server over stdio (default) or HTTP (``--http``); returns the exit code."""
    parser = argparse.ArgumentParser(prog="python -m src.mcp_server", description=__doc__)
    parser.add_argument("--http", action="store_true", help="Streamable HTTP instead of stdio")
    parser.add_argument("--port", type=int, help="HTTP port (default: mcp.port in config.yaml)")
    parser.add_argument(
        "--host",
        help="HTTP address (default: mcp.host, 127.0.0.1). The Docker image uses 0.0.0.0 "
        "inside the container and publishes the port on the host's 127.0.0.1 only.",
    )
    args = parser.parse_args(argv)

    # Claude Desktop starts the server from its own folder; paths are relative to the project.
    os.chdir(PROJECT_ROOT)
    try:
        settings = get_settings()
    except ConfigError as exc:
        print(f"Finnie MCP server: {exc}", file=sys.stderr)
        return 2
    configure_logging(settings.app.log_level, fmt="text", stream=sys.stderr)

    from src.mcp_server.server import build_server

    if not args.http:
        build_server().run("stdio")
        return 0
    return serve_http(build_server, settings, port=args.port, host=args.host)


def serve_http(
    build_server: Callable[[], MCPServer],
    settings: Settings,
    *,
    port: int | None = None,
    host: str | None = None,
) -> int:
    """Serve Streamable HTTP with uvicorn; exits with 2 if MCP_API_TOKEN isn't usable."""
    import uvicorn

    from src.mcp_server.http import TokenError, build_http_app

    overrides = {k: v for k, v in {"port": port, "host": host}.items() if v is not None}
    if overrides:
        settings = settings.model_copy(update={"mcp": settings.mcp.model_copy(update=overrides)})
    try:
        app = build_http_app(build_server(), settings)
    except TokenError as exc:
        print(f"Finnie MCP server: {exc}", file=sys.stderr)
        return 2
    mcp = settings.mcp
    print(
        f"Finnie MCP server on http://{mcp.host}:{mcp.port}{mcp.path} (bearer token required)",
        file=sys.stderr,
    )
    uvicorn.run(app, host=mcp.host, port=mcp.port, log_level="warning")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
