"""Helpers shared by the MCP server tests."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import anyio
from mcp import Client

from src.mcp_server.server import Services, build_server

TOKEN = "test-token-" + "x" * 32


def with_client[T](services: Services, use: Callable[[Client], Awaitable[T]]) -> T:
    """Run ``use`` against an in-memory client of a server built on ``services``."""

    async def main() -> T:
        async with Client(build_server(services)) as client:
            return await use(client)

    return anyio.run(main)
