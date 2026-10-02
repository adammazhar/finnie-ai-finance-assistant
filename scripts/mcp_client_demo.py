"""Demo client for Finnie's MCP server over Streamable HTTP.

Start the server first (in another terminal):

    python -m src.mcp_server --http

Then:

    python scripts/mcp_client_demo.py                 # tax-account tool (works offline)
    python scripts/mcp_client_demo.py --ticker AAPL   # also a live quote

It shows:
1. a request with no token, rejected with 401;
2. a request with a wrong token, rejected with 401;
3. with the token from .env (MCP_API_TOKEN): listing the tools and calling one.

The token is read from .env and never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from src.core.config import get_settings
from src.mcp_server.http import TokenError, api_token, url

# A minimal MCP request, the same shape any client sends first.
PING = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "MCP-Protocol-Version": "2026-07-28",
}


async def raw_request(endpoint: str, token: str | None) -> httpx2.Response:
    headers = dict(HEADERS)
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx2.AsyncClient(timeout=10) as http:
        return await http.post(endpoint, json=PING, headers=headers)


def show_rejection(label: str, response: httpx2.Response) -> bool:
    print(f"\n{label}")
    challenge = response.headers.get("www-authenticate")
    print(f"  HTTP {response.status_code}  WWW-Authenticate: {challenge}")
    print(f"  {response.text}")
    return response.status_code == 401


def structured(result: Any) -> str:
    if result.structured_content is not None:
        return json.dumps(result.structured_content, indent=2)
    return "\n".join(getattr(c, "text", str(c)) for c in result.content)


async def authorized(endpoint: str, token: str, ticker: str | None) -> None:
    http = httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"}, timeout=60)
    async with http, Client(streamable_http_client(endpoint, http_client=http)) as client:
        print(f"\n3. With the token: connected (protocol {client.protocol_version})")
        tools = await client.list_tools()
        print(f"  {len(tools.tools)} tools:")
        for tool in tools.tools:
            print(f"   - {tool.name}: {tool.title or ''}")

        print('\n  call explain_tax_account(account_type="roth_ira"):')
        result = await client.call_tool("explain_tax_account", {"account_type": "roth_ira"})
        print("  " + structured(result).replace("\n", "\n  "))

        if ticker:
            print(f'\n  call get_stock_quote(ticker="{ticker}"):')
            result = await client.call_tool("get_stock_quote", {"ticker": ticker})
            print("  " + structured(result).replace("\n", "\n  "))


async def main(ticker: str | None) -> int:
    settings = get_settings()
    try:
        token = api_token(settings)
    except TokenError as exc:
        print(exc, file=sys.stderr)
        return 2
    endpoint = url(settings)
    print(f"Finnie MCP server: {endpoint}")
    try:
        ok = show_rejection("1. No token:", await raw_request(endpoint, None))
        ok &= show_rejection("2. Wrong token:", await raw_request(endpoint, "not-the-token"))
    except httpx2.ConnectError:
        print("Can't connect. Start the server first: python -m src.mcp_server --http")
        return 1
    if not ok:
        print("\nExpected 401 for both; the server isn't enforcing the token.", file=sys.stderr)
        return 1
    await authorized(endpoint, token, ticker)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ticker", help="also call get_stock_quote for this ticker")
    sys.exit(asyncio.run(main(parser.parse_args().ticker)))
