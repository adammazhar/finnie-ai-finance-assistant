"""MCP tool latency over Streamable HTTP: python scripts/bench_mcp.py [--runs N].

Starts the HTTP server in this process on a free loopback port, with a throwaway token
(nothing is read from or written to .env), then measures through a real MCP client:

1. how long after the server starts a knowledge base search first answers (the embedding
   model and index load in the background from startup);
2. once that's done, each tool's first call (market data fetched) and the median of the
   next N calls (market data from the cache). Uses live market data.
"""

from __future__ import annotations

import argparse
import secrets
import socket
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anyio
import httpx2
import uvicorn
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from pydantic import SecretStr

from src.core.config import get_settings
from src.mcp_server.http import build_http_app, url
from src.mcp_server.server import build_server

CALLS: list[tuple[str, dict[str, Any]]] = [
    ("explain_tax_account", {"account_type": "roth_ira"}),
    ("project_financial_goal", {"target_amount": 50000, "years": 10, "monthly_contribution": 300}),
    ("search_financial_knowledge", {"query": "How do expense ratios work?"}),
    ("get_stock_quote", {"ticker": "VTI"}),
    ("get_market_overview", {}),
    (
        "analyze_portfolio",
        {"holdings": [{"ticker": "VTI", "shares": 10}, {"ticker": "BND", "shares": 20}]},
    ),
]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def measure(
    endpoint: str, token: str, runs: int, started_at: float
) -> tuple[float, list[tuple[str, float, float]]]:
    rows = []
    http = httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"}, timeout=120)
    async with http, Client(streamable_http_client(endpoint, http_client=http)) as client:
        await client.call_tool("search_financial_knowledge", {"query": "What is an ETF?"})
        ready = (time.perf_counter() - started_at) * 1000
        for name, args in CALLS:
            times = []
            for _ in range(runs + 1):
                started = time.perf_counter()
                result = await client.call_tool(name, args)
                times.append((time.perf_counter() - started) * 1000)
                if result.is_error:
                    raise SystemExit(f"{name} failed: {result.content}")
            rows.append((name, times[0], statistics.median(times[1:])))
    return ready, rows


def main(runs: int) -> None:
    token = secrets.token_urlsafe(32)
    settings = get_settings()
    settings = settings.model_copy(
        update={
            "mcp_api_token": SecretStr(token),
            "mcp": settings.mcp.model_copy(update={"port": free_port()}),
        }
    )
    started_at = time.perf_counter()  # the knowledge base warm-up starts in build_server()
    server = uvicorn.Server(
        uvicorn.Config(
            build_http_app(build_server(), settings),
            host=settings.mcp.host,
            port=settings.mcp.port,
            log_level="warning",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    try:
        ready, rows = anyio.run(measure, url(settings), token, runs, started_at)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    print(f"First search answered {ready / 1000:.1f} s after the server started.\n")
    print(f"| Tool (server warm) | First call | Median of next {runs} |")
    print("|---|---|---|")
    for name, cold, warm in rows:
        print(f"| `{name}` | {cold:,.0f} ms | {warm:,.0f} ms |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=int, default=10)
    main(parser.parse_args().runs)
