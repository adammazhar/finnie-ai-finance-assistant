"""Streamable HTTP transport: a real uvicorn server on loopback, behind the bearer token."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from typing import Any

import anyio
import httpx2
import pytest
import uvicorn
from mcp import Client, MCPError
from mcp.client.streamable_http import streamable_http_client

from src.core.config import MCPConfig
from src.mcp_server.http import TokenError, api_token, build_http_app, transport_security, url
from src.mcp_server.server import build_server
from tests.unit.mcp_server.support import TOKEN

LIST_TOOLS = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {
        "_meta": {
            "io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientCapabilities": {},
        }
    },
}
LEGACY_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "older-client", "version": "1"},
    },
}
BASE_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "MCP-Protocol-Version": "2026-07-28",
    "Mcp-Method": "tools/list",
}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def endpoint(make_settings, services) -> Iterator[str]:
    settings = make_settings(mcp_api_token=TOKEN)
    settings = settings.model_copy(
        update={"mcp": settings.mcp.model_copy(update={"port": free_port()})}
    )
    app = build_http_app(build_server(services), settings)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=settings.mcp.port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        assert time.monotonic() < deadline, "server didn't start"
        time.sleep(0.02)
    yield url(settings)
    server.should_exit = True
    thread.join(timeout=10)


def post(
    endpoint: str,
    *,
    authorization: str | None = None,
    body: dict[str, Any] = LIST_TOOLS,
    **headers: str,
) -> httpx2.Response:
    sent = {**BASE_HEADERS, **headers}
    if authorization is not None:
        sent["Authorization"] = authorization
    return httpx2.post(endpoint, content=json.dumps(body), headers=sent, timeout=10)


# ---- token -----------------------------------------------------------------------------


def test_api_token_is_required(make_settings):
    with pytest.raises(TokenError, match="isn't set"):
        api_token(make_settings())
    with pytest.raises(TokenError, match="isn't set"):
        api_token(make_settings(mcp_api_token="   "))
    with pytest.raises(TokenError, match="too short"):
        api_token(make_settings(mcp_api_token="short"))
    assert api_token(make_settings(mcp_api_token=f" {TOKEN} ")) == TOKEN


def test_server_refuses_to_start_without_a_token(make_settings, services):
    with pytest.raises(TokenError):
        build_http_app(build_server(services), make_settings())


def test_transport_security_allows_only_localhost():
    security = transport_security(MCPConfig())
    assert security.enable_dns_rebinding_protection
    assert security.allowed_hosts == ["127.0.0.1:*", "[::1]:*", "localhost:*"]
    assert security.allowed_origins == [
        "http://127.0.0.1:*",
        "http://[::1]:*",
        "http://localhost:*",
    ]


# ---- 401 -------------------------------------------------------------------------------


def test_request_without_a_token_is_rejected_with_401(endpoint):
    response = post(endpoint)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == 'Bearer realm="finnie"'
    assert response.json()["error"] == "missing_token"


@pytest.mark.parametrize(
    "authorization",
    [
        "Bearer wrong-token",
        f"Bearer {TOKEN}x",
        f"Bearer {TOKEN[:-1]}",
        f"Basic {TOKEN}",
        TOKEN,
        "Bearer",
    ],
)
def test_request_with_an_invalid_token_is_rejected_with_401(endpoint, authorization):
    response = post(endpoint, authorization=authorization)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == 'Bearer realm="finnie", error="invalid_token"'
    assert TOKEN not in response.text


def test_get_without_a_token_is_401_not_405(endpoint):
    assert httpx2.get(endpoint, timeout=10).status_code == 401


def test_rejections_are_logged_without_the_token(endpoint, caplog):
    post(endpoint, authorization="Bearer wrong-token")
    assert "Rejected MCP request: invalid_token" in caplog.text
    assert "wrong-token" not in caplog.text


# ---- with the token ----------------------------------------------------------------------


def test_valid_token_lists_tools(endpoint):
    response = post(endpoint, authorization=f"Bearer {TOKEN}")
    assert response.status_code == 200
    names = {tool["name"] for tool in response.json()["result"]["tools"]}
    assert "get_stock_quote" in names


def test_scheme_is_case_insensitive(endpoint):
    assert post(endpoint, authorization=f"bearer {TOKEN}").status_code == 200


@pytest.mark.parametrize("method", ["GET", "DELETE", "PUT"])
def test_only_post_is_supported(endpoint, method):
    response = httpx2.request(
        method, endpoint, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=10
    )
    assert response.status_code == 405
    assert response.headers["allow"] == "POST"


def test_older_clients_can_still_initialize(endpoint):
    response = post(
        endpoint,
        authorization=f"Bearer {TOKEN}",
        body=LEGACY_INITIALIZE,
        **{"MCP-Protocol-Version": "2025-06-18", "Mcp-Method": "initialize"},
    )
    assert response.status_code == 200
    assert response.json()["result"]["serverInfo"]["name"] == "Finnie"


# ---- DNS-rebinding protection (needs the token to get this far) ---------------------------


def test_foreign_origin_is_rejected(endpoint):
    response = post(endpoint, authorization=f"Bearer {TOKEN}", Origin="https://evil.example")
    assert response.status_code == 403


def test_local_origin_is_allowed(endpoint):
    response = post(endpoint, authorization=f"Bearer {TOKEN}", Origin="http://localhost:3000")
    assert response.status_code == 200


def test_foreign_host_is_rejected(endpoint):
    response = post(endpoint, authorization=f"Bearer {TOKEN}", Host="evil.example:8765")
    assert response.status_code == 421


# ---- a real MCP client ------------------------------------------------------------------------


def test_mcp_client_with_token(endpoint):
    async def main() -> tuple[set[str], dict[str, Any]]:
        http = httpx2.AsyncClient(headers={"Authorization": f"Bearer {TOKEN}"}, timeout=30)
        async with http, Client(streamable_http_client(endpoint, http_client=http)) as client:
            tools = await client.list_tools()
            result = await client.call_tool("get_stock_quote", {"ticker": "VTI"})
            return {t.name for t in tools.tools}, result.structured_content

    names, quote = anyio.run(main)
    assert len(names) == 6
    assert quote["ticker"] == "VTI" and quote["price"] == 300.0


def test_mcp_client_without_token_fails(endpoint, caplog):
    async def main() -> None:
        async with Client(endpoint) as client:
            await client.list_tools()

    with pytest.raises(BaseExceptionGroup) as raised:
        anyio.run(main)
    assert raised.group_contains(MCPError, depth=None)
    assert "Rejected MCP request: missing_token" in caplog.text
